from pathlib import Path
from datetime import datetime
from itertools import groupby
from collections import namedtuple

import numpy as np
from stixpy.calibration.visibility import (
    calibrate_visibility,
    create_meta_pixels,
    create_visibility,
)
from stixpy.coordinates.transforms import get_hpc_info
from stixpy.net.client import STIXClient
from stixpy.product import Product as STIXPYProduct
from sunpy.coordinates import HeliographicStonyhurst, Helioprojective, SphericalScreen
from sunpy.map import make_fitswcs_header
from sunpy.net import attrs as a
from sunpy.time import TimeRange
from xrayvision.clean import vis_clean

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.representation import CartesianRepresentation
from astropy.io import fits
from astropy.table import Column, QTable
from astropy.time import Time

from stixcore.config.config import CONFIG
from stixcore.ephemeris.manager import Spice
from stixcore.products.level3.flarelistproduct import PeakPreviewImage
from stixcore.products.level3.processing import stx_estimate_flare_location
from stixcore.products.product import CountDataMixin, GenericProduct, L2Mixin
from stixcore.soop.manager import SOOPManager
from stixcore.time import SCETime, SCETimeRange
from stixcore.util.logging import get_logger
from stixcore.util.util import url_to_path

from stixpy.map.stix import STIXMap  # noqa

__all__ = [
    "FlarelistSDC",
    "FlarePositionMixin",
    "FlareSOOPMixin",
    "FlareList",
    "FlarelistSDCLocation",
    "FlarelistSDCLocationImage",
    "FlarePeakPreviewMixin",
    "FlarelistSC",
    "FlarelistSCLocation",
    "FlarelistSCLocationImage",
    "add_distance_normalized_flux",
]

logger = get_logger(__name__)


def make_stix_fitswcs_header(data, flare_position, *, scale, exposure, rotation_angle, energy_range):
    """Create a FITS WCS header for the given data."""
    header = make_fitswcs_header(
        data,
        flare_position,
        telescope="STIX",
        observatory="Solar Orbiter",
        scale=scale,
        exposure=exposure,
        rotation_angle=rotation_angle,
    )

    if energy_range is not None and header["wcsaxes"] == 2:
        # add energy range to WCS header
        # as a fake 3rd axis
        header["wcsaxes"] = 3
        header["CRVAL3"] = energy_range.mean().value
        header["CRPIX3"] = 1
        header["CDELT3"] = energy_range[1].value - energy_range[0].value
        header["CUNIT3"] = str(energy_range[0].unit)
        header["CTYPE3"] = "ENER"
        header["NAXIS"] = 3
        header["NAXIS3"] = 1

    return header


#: Which per-flare distance column each flux column is normalized by. LC peak fluxes use the
#: flare-time SOLO distance; the quiet-period background fluxes use the background CPD's own time.
_FLUX_DISTANCE_COLS = {
    "lc_peak_flux": "solo_sun_distance",
    "lc_bkg_peak_flux": "solo_sun_distance",
    "bkg_spec_flux": "bkg_solo_sun_distance",
    "bkg_spec_flux_ql": "bkg_solo_sun_distance",
}


def add_distance_normalized_flux(data, mapping=_FLUX_DISTANCE_COLS):
    """Add ``<col>_at_1au`` = ``<col> * (r/1AU)**2`` for each present flux column, using the
    per-flare distance in its mapped column (flux scales as 1/r**2).

    The existing flux columns are kept unchanged; the scale factor is dimensionless so the
    ``*_at_1au`` columns keep the flux unit (ct/s/keV/cm2). NaN distance -> NaN; a flux column
    (or its distance column) that is absent is skipped.
    """
    for col, dist_col in mapping.items():
        if col not in data.colnames or dist_col not in data.colnames:
            continue
        factor = (data[dist_col] / (1 * u.AU)).decompose().value ** 2  # (N,), dimensionless
        f = data[col]
        data[col + "_at_1au"] = f * (factor[:, None] if f.ndim == 2 else factor)
        data[
            col + "_at_1au"
        ].info.description = f"{col} scaled to what would be seen at 1 AU (x (r_solo/1AU)**2, r_solo from {dist_col})"


#: Per-flare result of :meth:`FlarePositionMixin.add_flare_position`. One record per flare row;
#: skipped/failed flares use :func:`_empty_flare_position` (NaN geometry, ``peak_time`` placeholder).
FlarePositionResult = namedtuple(
    "FlarePositionResult",
    [
        "anc_path",
        "cpd_path",
        "status",
        "message",
        "flare_x",  # HGS cartesian, km
        "flare_y",
        "flare_z",
        "solo_time",  # location time center
        "duration",  # location window length
        "solo_x",  # SOLO HGS cartesian, km
        "solo_y",
        "solo_z",
        "rcr_at_peak",
        "sidelobe",
        "min_exposure",  # CPD per-bin exposure ds over the flare window (u.ds)
        "max_exposure",
    ],
)


def _empty_flare_position(solo_time, **overrides):
    """A :class:`FlarePositionResult` for a skipped/failed flare.

    NaN geometry and zero window, with ``solo_time`` kept as the per-row time placeholder (it
    must stay a valid `~astropy.time.Time` for the downstream coordinate/Spice handling).
    ``overrides`` set the few fields a given skip site knows, e.g. ``message`` / ``anc_path`` /
    ``cpd_path``.
    """
    base = dict(
        anc_path="",
        cpd_path="",
        status=False,
        message="",
        flare_x=np.nan * u.km,
        flare_y=np.nan * u.km,
        flare_z=np.nan * u.km,
        solo_time=solo_time,
        duration=0 * u.s,
        solo_x=np.nan * u.km,
        solo_y=np.nan * u.km,
        solo_z=np.nan * u.km,
        rcr_at_peak=0,
        sidelobe=np.nan,
        min_exposure=np.nan * u.ds,
        max_exposure=np.nan * u.ds,
    )
    base.update(overrides)
    return FlarePositionResult(**base)


def cpd_timedel_range(times, timedels, start, end):
    """Min and max CPD time-step ``ds`` (``timedel``) for bins overlapping ``[start, end]``.

    Uses the same half-bin overlap test as the peak-window selection so partially covered
    windows still contribute. Returns ``(min_exposure, max_exposure)`` as
    `~astropy.units.Quantity` in deciseconds (``u.ds``, STIX's native ``timedel`` unit);
    ``(NaN ds, NaN ds)`` if no bin overlaps.
    """
    half = timedels / 2
    mask = (times + half >= start) & (times - half <= end)
    sel = timedels[mask]
    if len(sel) == 0:
        return np.nan * u.ds, np.nan * u.ds
    return sel.min().to(u.ds), sel.max().to(u.ds)


class _SerializeMixin:
    """No-op chain terminator for on_serialize/on_deserialize.

    Functional mixins inherit from this so super() calls always land safely
    instead of hitting object and raising AttributeError.
    """

    def on_serialize(self, data):
        pass

    def on_deserialize(self, data, **kwargs):
        pass


class FlarePositionMixin(_SerializeMixin):
    """Mixin adding a STIX-derived flare location to a flare-list product.

    For every flare it selects a compressed-pixel-data (CPD) file, estimates the source
    position by back-projection imaging (`~stixcore.products.level3.processing.stx_estimate_flare_location`),
    and stores the location (Heliographic Stonyhurst / Helioprojective), the Solar Orbiter
    position and distance, an imaging-quality metric and related timing columns. On
    serialization the location columns are converted to ICRS (and back on deserialization) so
    they survive the FITS round-trip.

    See :doc:`/products/flarelist` for the CPD-selection and imaging details. Used by
    `~stixcore.products.level3.flarelist.FlarelistSDCLocation` (and the image product built on
    it).
    """

    @classmethod
    def add_flare_position(
        cls,
        data,
        fido_client: STIXClient,
        *,
        filter_function=lambda x: True,
        peak_time_colname="peak_UTC",
        start_time_colname="start_UTC",
        end_time_colname="end_UTC",
        location_time_colname="location_time_UTC",
        keep_all_flares=True,
        month=None,
    ):
        """Estimate and add the flare location columns for every flare in ``data``.

        For each flare passing ``filter_function`` the daily ancillary ephemeris and the
        science CPD file(s) covering ``[start, end]`` are looked up; the best CPD is scored and
        selected, a constant-``rcr`` time window around the peak is chosen, and the location is
        estimated by back-projection imaging. See :doc:`/products/flarelist` for the selection
        and imaging parameters.

        Parameters
        ----------
        data : `~astropy.table.QTable`
            The flare list; location columns are added in place.
        fido_client : `~stixpy.net.client.STIXClient`
            Client used to search for the ephemeris and CPD files.
        filter_function : callable, optional
            ``row -> bool`` deciding which flares to process (default: all).
        peak_time_colname, start_time_colname, end_time_colname, location_time_colname : str
            Names of the time columns to read / write.
        keep_all_flares : bool, optional
            If False, flares that did not pass ``filter_function`` are removed from ``data``.
        month : optional
            Month being processed, used for logging only.
        """
        position_results = []

        to_remove = []
        pass_filter = 0
        no_ephemeris = 0
        no_cpd = 0
        many_cpd = 0
        one_cpd = 0
        total_flares = len(data)

        day_asp_ephemeris_cache = dict()

        for i, row in enumerate(data):
            peak_time = row[peak_time_colname]
            start_time = row[start_time_colname]
            end_time = row[end_time_colname]
            logger.info(f"Processing flare {i}/{len(data)} at time {start_time} : {end_time} (peak at {peak_time})")
            if filter_function(row):  # and i < 60:
                pass_filter += 1
                day = peak_time.to_datetime().date()

                if day in day_asp_ephemeris_cache:
                    anc_res = day_asp_ephemeris_cache[day]
                else:
                    anc_res = fido_client.search(
                        a.Time(peak_time, peak_time), a.Instrument.stix, a.stix.DataProduct.asp_ephemeris
                    )
                    if anc_res:
                        anc_res.filter_for_latest_version()
                    url_to_path(anc_res)
                    day_asp_ephemeris_cache[day] = anc_res

                if len(anc_res) < 1:
                    logger.warning(f"No ephemeris data found for flare at time {start_time} : {end_time}")
                    no_ephemeris += 1
                    position_results.append(_empty_flare_position(peak_time, message="no ephemeris data found"))
                    continue
                _anc_path = str(anc_res["path"][0])

                # widen only the FIDO search window near midnight; keep the true flare start_time
                search_start = start_time - 2 * u.hour if start_time.datetime.hour < 2 else start_time
                cpd_res = fido_client.search(
                    a.Time(search_start, end_time), a.Instrument.stix, a.stix.DataProduct.sci_xray_cpd
                )
                if cpd_res:
                    cpd_res.filter_for_latest_version()
                url_to_path(cpd_res)

                if len(cpd_res) < 1:
                    logger.warning(f"No CPD data found for flare at time {search_start} : {end_time}")
                    no_cpd += 1
                    position_results.append(
                        _empty_flare_position(peak_time, anc_path=_anc_path, message="no CPD data found")
                    )
                    continue
                if len(cpd_res) > 1:
                    logger.debug(f"Many CPD data found for flare at time {search_start} : {end_time}")
                    # select the best available CPD data file (full product read per candidate)
                    cpd_res["inc_peak"] = False  # flare peak time inside the file
                    cpd_res["inc_flare"] = 0.0  # % of the flare duration covered by the file
                    cpd_res["min_dt"] = np.inf  # min time resolution (ds) during the flare; inf if no overlap
                    cpd_res["ebins"] = 0  # number of energy bins (energy table)
                    many_cpd += 1

                    flare_dur_s = (end_time - start_time).sec  # flare duration in seconds
                    for ci, path in enumerate(cpd_res["path"]):
                        cpd = STIXPYProduct(Path(path))
                        # FIDO reports Start/End Time as strings; parse to Time for arithmetic
                        file_start = Time(cpd_res["Start Time"][ci])
                        file_end = Time(cpd_res["End Time"][ci])
                        cpd_res["inc_peak"][ci] = file_start <= peak_time <= file_end
                        # percentage of the flare [start_time, end_time] duration covered by the file
                        overlap_s = (min(file_end, end_time) - max(file_start, start_time)).sec
                        cpd_res["inc_flare"][ci] = 100.0 * max(0.0, overlap_s) / flare_dur_s if flare_dur_s > 0 else 0.0
                        # shortest time resolution among bins overlapping the flare (shorter is better)
                        dt_min, _ = cpd_timedel_range(cpd.data["time"], cpd.data["timedel"], start_time, end_time)
                        cpd_res["min_dt"][ci] = dt_min.to_value(u.ds) if np.isfinite(dt_min.value) else np.inf
                        cpd_res["ebins"][ci] = len(cpd.energies)

                    # best = peak inside, then most flare coverage, then shortest time resolution,
                    # then most energy bins. shorter min_dt is better, so sort on its negative.
                    # TODO: add more criteria to select the best CPD file
                    cpd_res["_neg_min_dt"] = -cpd_res["min_dt"]
                    cpd_res.sort(["inc_peak", "inc_flare", "_neg_min_dt", "ebins"], reverse=True)
                    # cpd_res.pprint()
                    best_cpd_idx = 0
                else:
                    one_cpd += 1
                    best_cpd_idx = 0
                _cpd_path = str(cpd_res["path"][best_cpd_idx])

                try:
                    stixpy_cpd = STIXPYProduct(Path(_cpd_path))

                    # CPD per-bin exposure (ds) over the full flare start..end window (partial overlap ok)
                    min_exposure, max_exposure = cpd_timedel_range(
                        stixpy_cpd.data["time"], stixpy_cpd.data["timedel"], start_time, end_time
                    )

                    time_range = TimeRange(max(peak_time - 20 * u.s, start_time), min(peak_time + 20 * u.s, end_time))
                    overlaps = calculate_overlap(stixpy_cpd.time_range, time_range)
                    if overlaps is None:
                        logger.warning(
                            f"CPD data does not cover time range around peak time {time_range.start} to {time_range.end}"
                        )
                        time_range = stixpy_cpd.time_range
                        contains_peak_time = False
                    else:
                        contains_peak_time = True
                        time_range = overlaps

                    _times = stixpy_cpd.data["time"]
                    _half_bin = stixpy_cpd.data["timedel"] / 2
                    mask = (_times + _half_bin >= time_range.start) & (_times - _half_bin <= time_range.end)
                    data_at_peak = stixpy_cpd.data[mask]
                    energy_range = [4, 16] * u.keV

                    if len(np.unique(data_at_peak["rcr"])) > 1:
                        logger.warning(
                            f"Multiple rcr values found for flare at time {time_range.start} : {time_range.end}"
                        )
                        # allow a larger time range for finding a constant rcr sequence
                        if contains_peak_time:
                            time_range = TimeRange(
                                max(peak_time - 40 * u.s, start_time), min(peak_time + 40 * u.s, end_time)
                            )
                            mask = (_times + _half_bin >= time_range.start) & (_times - _half_bin <= time_range.end)
                            data_at_peak = stixpy_cpd.data[mask]
                        length, start_idx, rcr = longest_constant_sequence(data_at_peak["rcr"].value)
                        time_range = TimeRange(
                            data_at_peak["time"][start_idx], data_at_peak["time"][start_idx + length - 1]
                        )
                        logger.info(
                            f"Using time range {time_range.start} to {time_range.end} for flare at around {peak_time} with constant rcr={rcr}"
                        )

                    rcr_at_peak = data_at_peak["rcr"].max()
                    if rcr_at_peak > 0:
                        energy_range = [4, 25] * u.keV

                    _, flare_loc, sidelobe, solo, img_time_range = stx_estimate_flare_location(
                        stixpy_cpd, time_range, energy_range
                    )

                    with SphericalScreen(solo, only_off_disk=True):
                        center_hgs = flare_loc.transform_to(
                            HeliographicStonyhurst(obstime=img_time_range.center)
                        ).cartesian
                        position_results.append(
                            FlarePositionResult(
                                anc_path=_anc_path,
                                cpd_path=_cpd_path,
                                status=True,
                                message="OK",
                                flare_x=center_hgs.x,
                                flare_y=center_hgs.y,
                                flare_z=center_hgs.z,
                                solo_time=img_time_range.center,
                                duration=img_time_range.seconds,
                                solo_x=solo.x,
                                solo_y=solo.y,
                                solo_z=solo.z,
                                rcr_at_peak=rcr_at_peak,
                                sidelobe=sidelobe,
                                min_exposure=min_exposure,
                                max_exposure=max_exposure,
                            )
                        )
                except Exception as e:
                    logger.warning(f"Error calculating flare position for flare at time {start_time} : {end_time}: {e}")
                    position_results.append(
                        _empty_flare_position(
                            peak_time, anc_path=_anc_path, cpd_path=_cpd_path, message=f"Error: {type(e)}"
                        )
                    )

            else:
                to_remove.append(i)
                position_results.append(
                    _empty_flare_position(peak_time, message="flare did not pass the filter function")
                )

        # transpose the per-flare records in one pass (a namedtuple is a tuple), field order must
        # match FlarePositionResult
        (
            anc_ephemeris_paths,
            cpd_paths,
            position_statuses,
            position_messages,
            flare_x,
            flare_y,
            flare_z,
            solo_times,
            duration,
            solo_x,
            solo_y,
            solo_z,
            rcr_at_peak,
            sidelobe,
            min_exposure,
            max_exposure,
        ) = zip(*position_results)
        solo_times = Time(solo_times)

        primer = fido_client.baseurl.replace(fido_client.datapath, "")
        primer = primer[7:] if primer.startswith("file://") else primer

        data["anc_ephemeris_path"] = [v.replace(primer, "") for v in anc_ephemeris_paths]
        data["anc_ephemeris_path"].info.description = "Path to the daily ancillary ephemeris file"

        data["cpd_path"] = [v.replace(primer, "") for v in cpd_paths]
        data["cpd_path"].info.description = "Path to the CPD file used for flare location estimation"

        data["_position_status"] = position_statuses
        data["_position_status"].info.description = "Status of the flare position calculation"

        data["_position_message"] = position_messages
        data["_position_message"].info.description = "Message describing the status of the flare position calculation"

        hgs_coords = SkyCoord(
            u.Quantity(flare_x),
            u.Quantity(flare_y),
            u.Quantity(flare_z),
            frame=HeliographicStonyhurst(obstime=solo_times),
            representation_type="cartesian",
        )

        solo_coords = SkyCoord(
            u.Quantity(solo_x),
            u.Quantity(solo_y),
            u.Quantity(solo_z),
            frame=HeliographicStonyhurst(obstime=solo_times),
            representation_type="cartesian",
        )

        # hgc_coords = hgs_coords.transform_to(HeliographicCarrington(obstime=solo_times, observer="Earth"))
        hp_coords = hgs_coords.transform_to(Helioprojective(obstime=solo_times, observer="Earth"))

        data["location_hgs"] = hgs_coords
        data["location_hgs"].info.description = "Flare location in Heliographic Stonyhurst coordinates"

        data["solo_location_hgs"] = solo_coords
        data["solo_location_hgs"].info.description = "SOLO location in Heliographic Stonyhurst coordinates"

        data["solo_sun_distance"] = solo_coords.cartesian.norm().to(u.km)
        data["solo_sun_distance"].info.description = "distance of Solar Orbiter to Sun center"

        # add 1-AU-normalized twins of the flux columns (flux ~ 1/r^2); keeps the originals
        add_distance_normalized_flux(data)

        data["sidelobes_ratio"] = sidelobe
        data["sidelobes_ratio"].info.description = "Ratio of sidelobes in the STIX image used to assess imaging quality"

        data["rcr_at_peak"] = rcr_at_peak
        data[
            "rcr_at_peak"
        ].info.description = "max rcr level at flare location estimation time range, > 0 attenuator in place"

        data["visible_from_earth"] = FlarePositionMixin.is_visible(hp_coords)
        data[
            "visible_from_earth"
        ].info.description = "Whether the flare location is visible from Earth (not occulted by the Sun)"

        data[location_time_colname] = solo_times
        data[location_time_colname].info.description = "time center used for flare location estimation in UTC"

        data["location_duration"] = duration
        data["location_duration"].info.description = "duration of the flare location estimation time range"

        data["min_exposure"] = u.Quantity(min_exposure)  # unit u.ds (deciseconds)
        data[
            "min_exposure"
        ].info.description = "minimum CPD per-bin exposure ds (timedel) over the flare start..end window"
        data["max_exposure"] = u.Quantity(max_exposure)
        data[
            "max_exposure"
        ].info.description = "maximum CPD per-bin exposure ds (timedel) over the flare start..end window"

        (
            time_shift,
            disc_size,
        ) = zip(
            *[
                (Spice.instance.get_earth_solo_time_shift(date=scet), Spice.instance.get_sun_disc_size(date=scet))
                for t in solo_times
                for scet in (Spice.instance.datetime_to_scet(t),)
            ]
        )

        data["time_shift"] = time_shift
        data["time_shift"].info.description = "Time(Sun to Earth) - Time(Sun to S/C)"

        data["sun_disc_size"] = disc_size
        data["sun_disc_size"].info.description = "Apparent photospheric solar radius"

        if not keep_all_flares:
            data.remove_rows(to_remove)

        logger.info(
            f"Flare position calculated for month {month} with {total_flares} flares, "
            f"passed filter: {pass_filter} no ephemeris data found for {no_ephemeris} "
            f"flares, no CPD data found for {no_cpd} flares, many CPD data found for "
            f"{many_cpd} flares, one CPD data found for {one_cpd} flares."
            f"finally {len(data) - len(to_remove)} flare locations found"
        )

    def on_serialize(self, data):
        logger.warning(
            "FlarePositionMixin on_serialize called, transforming location columns to ICRS for serialization"
        )

        if "location_hgs" in data.colnames:
            icrs = data["location_hgs"].icrs
            icrs_coord = SkyCoord(icrs.ra, icrs.dec, icrs.distance, frame="icrs")
            col_idx = data.colnames.index("location_hgs")
            data.remove_column("location_hgs")
            data.add_column(icrs_coord, name="location_icrs", index=col_idx)
        if "solo_location_hgs" in data.colnames:
            icrs = data["solo_location_hgs"].icrs
            icrs_coord = SkyCoord(icrs.ra, icrs.dec, icrs.distance, frame="icrs")
            col_idx = data.colnames.index("solo_location_hgs")
            data.remove_column("solo_location_hgs")
            data.add_column(icrs_coord, name="solo_location_icrs", index=col_idx)
        super().on_serialize(data)

    def on_deserialize(self, data, *, location_time_colname=None, **kwargs):
        logger.warning(
            "FlarePositionMixin on_deserialize called, transforming location columns back to heliographic coordinates"
        )
        time_col = location_time_colname or self.location_time_colname
        if time_col not in data.colnames:
            logger.warning(f"on_deserialize: column '{time_col}' not found, skipping location transform")
        else:
            obstime = Time(data[time_col])
            if "location_icrs" in data.colnames:
                data["location_hgs"] = data["location_icrs"].transform_to(HeliographicStonyhurst(obstime=obstime))
            if "solo_location_icrs" in data.colnames:
                data["solo_location_hgs"] = data["solo_location_icrs"].transform_to(
                    HeliographicStonyhurst(obstime=obstime)
                )

        super().on_deserialize(data, **kwargs)

    @classmethod
    def is_visible(cls, coord):
        """
        Returns whether the coordinate is on the visible side of the Sun.
        This function is a modified version of PR#7118
        """

        coord = coord.make_3d()
        data = coord.cartesian
        data_to_sun = coord.observer.radius * CartesianRepresentation(1, 0, 0) - data

        is_behind = data.x < 0
        # print(data.x.to(u.AU))
        is_beyond_limb = np.sqrt(1 - (data.x / data.norm()) ** 2) > coord.rsun / coord.observer.radius
        # is_above_surface = data_to_sun.norm() >= coord.rsun

        is_on_near_side = data.dot(data_to_sun) >= 0

        return is_behind | is_beyond_limb | (is_on_near_side)


class FlareSOOPMixin(_SerializeMixin):
    """Mixin adding the Solar Orbiter observing-campaign (SOOP) columns to a flare list.

    For each flare it queries `~stixcore.soop.manager.SOOPManager` for the campaign active at
    the flare peak and stores its encoded type, instance id and name (``soop_encoded_type``,
    ``soop_id``, ``soop_type``). Used by `~stixcore.products.level3.flarelist.FlarelistSDC`.
    """

    @classmethod
    def add_soop(
        self, data, *, peak_time_colname="peak_UTC", start_time_colname="start_UTC", end_time_colname="end_UTC"
    ):
        """Add the SOOP campaign columns for every flare in ``data`` (in place).

        Parameters
        ----------
        data : `~astropy.table.QTable`
            The flare list.
        peak_time_colname, start_time_colname, end_time_colname : str
            Names of the time columns; the campaign is looked up at ``peak_time_colname``.
        """
        soop_encoded_type = list()
        soop_id = list()
        soop_type = list()

        for row in data:
            soops = SOOPManager.instance.find_soops(start=row[peak_time_colname])
            if soops:
                soop = soops[0]
                soop_encoded_type.append(soop.encodedSoopType)
                soop_id.append(soop.soopInstanceId)
                soop_type.append(soop.soopType)
            else:
                soop_encoded_type.append(None)
                soop_id.append(None)
                soop_type.append(None)

        data["soop_encoded_type"] = Column(soop_encoded_type, dtype=str, description="campaign ID")
        data["soop_id"] = Column(soop_id, dtype=str, description="SOOP ID")
        data["soop_type"] = Column(soop_type, dtype=str, description="name of the SOOP campaign")

    # def on_serialize(self, data):
    #     logger.info("FlareSOOPMixin on_serialize called, but no special handling implemented for SOOP data")
    #     super().on_serialize(data)

    # def on_deserialize(self, data, **kwargs):
    #     logger.info("FlareSOOPMixin on_deserialize called, but no special handling implemented for SOOP data")
    #     super().on_deserialize(data, **kwargs)


class FlarePeakPreviewMixin:
    """Mixin class to add peak preview images to flare list products.
    This class provides a method to generate and add peak preview images
    to the flare list data. The images are generated based on the
    flare's peak time, start time, and end time, using the STIXPy library
    for visibility calculations and image reconstruction.
    The generated images are stored in the 'peak_preview_path' column of the data.
    The method also updates the status and message columns to indicate
    the success or failure of the image generation process.

    Currently the images are created for two energy ranges: 4-20 keV and 20-120 keV.
    """

    @classmethod
    def add_peak_preview(
        cls,
        data,
        energies,
        parent,
        fido_client: STIXClient,
        img_processor,
        *,
        peak_time_colname="peak_UTC",
        start_time_colname="start_UTC",
        end_time_colname="end_UTC",
        anc_ephemeris_path_colname="anc_ephemeris_path",
        cpd_path_colname="cpd_path",
        product_name_suffix="fl",
        keep_all_flares=True,
        month=None,
    ):
        """Reconstruct and attach per-flare peak-preview CLEAN images (4-20 and 20-120 keV) for
        each flare in ``data``, using its selected CPD file, writing one
        `~stixcore.products.level3.flarelistproduct.PeakPreviewImage` product per flare."""
        data["peak_preview_path"] = Column(" " * 500, dtype=str, description="TDB")
        data["preview_start_UTC"] = [Time(d, format="isot", scale="utc") for d in data[peak_time_colname]]
        data["preview_end_UTC"] = [Time(d, format="isot", scale="utc") for d in data[peak_time_colname]]
        data["_peak_preview_status"] = Column(False, dtype=bool, description="TDB")
        data["_peak_preview_message"] = Column(" " * 500, dtype=str, description="TDB")
        to_remove = []
        products = []
        images = 0

        for i, row in enumerate(data):
            peak_time = row[peak_time_colname]
            row[start_time_colname]
            row[end_time_colname]

            anc_ephemeris_path = Path(row[anc_ephemeris_path_colname])
            cpd_path = Path(row[cpd_path_colname])

            status = False
            message = ""

            peak_preview_start = row[peak_time_colname]
            peak_preview_end = row[peak_time_colname]

            if anc_ephemeris_path.exists() and cpd_path.exists():
                try:
                    status = True
                    # do the imaging with stixpy

                    preview_data = data[i : i + 1]
                    del preview_data["peak_preview_path"]
                    del preview_data["_peak_preview_status"]
                    del preview_data["_peak_preview_message"]

                    peak_preview_start = row[peak_time_colname] - 10 * u.s
                    peak_preview_end = row[peak_time_colname] + 10 * u.s

                    preview_data["preview_start_UTC"] = peak_preview_start
                    preview_data["preview_end_UTC"] = peak_preview_end

                    cpd_sci = STIXPYProduct(cpd_path)
                    time_range_sci = [peak_preview_start, peak_preview_end]
                    maps = []
                    for energy_range in [[4, 20], [20, 120]] * u.keV:
                        # flare_position = preview_data['flare_position'][0]
                        # flare_position = [0, 0] * u.arcsec
                        comments = []
                        helio_frame = Helioprojective(observer="earth", obstime=peak_time)
                        flare_position = SkyCoord(0 * u.deg, 0 * u.deg, frame=helio_frame)

                        meta_pixels_sci = create_meta_pixels(
                            cpd_sci,
                            time_range=time_range_sci,
                            energy_range=energy_range,
                            flare_location=flare_position,
                            no_shadowing=True,
                        )
                        vis = create_visibility(meta_pixels_sci)
                        cal_vis = calibrate_visibility(vis, flare_location=flare_position)
                        isc_10_3 = [
                            3,
                            20,
                            22,
                            16,
                            14,
                            32,
                            21,
                            26,
                            4,
                            24,
                            8,
                            28,
                            15,
                            27,
                            31,
                            6,
                            30,
                            2,
                            25,
                            5,
                            23,
                            7,
                            29,
                            1,
                        ]
                        col_idx = np.argwhere(np.isin(cal_vis.meta["isc"], isc_10_3)).ravel()
                        cal_vis.meta["offset"] = flare_position
                        vis10_3 = cal_vis[col_idx]

                        imsize = [129, 129] * u.pixel  # number of pixels of the map to reconstruct
                        pixel = [2, 2] * u.arcsec / u.pixel  # pixel size in arcsec

                        vis_tr = TimeRange(vis.meta["time_range"])
                        roll, solo_xyz, pointing = get_hpc_info(vis_tr.start, vis_tr.end)
                        solo = HeliographicStonyhurst(*solo_xyz, obstime=vis_tr.center, representation_type="cartesian")

                        clean_map, model_map, resid_map = vis_clean(
                            vis10_3, imsize, pixel_size=pixel, gain=0.1, niter=200, clean_beam_width=20 * u.arcsec
                        )
                        comments.append(f"clean map with {len(col_idx)} visibilities")
                        comments.append(f"clean gain: {0.1}, niter: {200}, clean beam width: {20 * u.arcsec}")
                        comments.append(f"det: {', '.join(sorted(vis10_3.meta['vis_labels']))}")

                        map_with_erange = clean_map.data[np.newaxis, ...]
                        map_with_erange[0, :, :] = clean_map.data
                        fp_hp = flare_position.transform_to(Helioprojective(obstime=vis_tr.center, observer=solo))
                        header = make_stix_fitswcs_header(
                            map_with_erange,
                            fp_hp,
                            scale=pixel,
                            exposure=vis_tr.seconds,
                            rotation_angle=90 * u.deg + roll,
                            energy_range=energy_range,
                        )

                        header = fits.Header(header)
                        # Add comments
                        [header.add_comment(com) for com in comments]

                        header["IMG_METH"] = ("clean", "STIX image reconstruction method used")

                        maps.append((map_with_erange, header))

                    ppi = PeakPreviewImage(
                        control=QTable(),
                        data=preview_data,
                        month=month,
                        energy=energies,
                        maps=maps,
                        product_name_suffix=product_name_suffix,
                        parents=[parent, anc_ephemeris_path.name, cpd_path.name],
                    )

                    for f in img_processor.write_fits(ppi):
                        products.append(f)
                        images += len(ppi.maps)
                    message = "OK"
                except Exception as e:
                    logger.error(e, stack_info=True)
                    status = False
                    message = str(e)

            data[i]["preview_start_UTC"] = peak_preview_start
            data[i]["preview_end_UTC"] = peak_preview_end
            data[i]["peak_preview_path"] = "test"
            data[i]["_peak_preview_status"] = status
            data[i]["_peak_preview_message"] = message

        if not keep_all_flares:
            data.remove_rows(to_remove)

        logger.info(
            f"Flare images created for month {month} with {len(data)} flares, "
            f"{len(products)} peak previews created, with total {images} images"
        )

        return products


class FlareList(CountDataMixin, GenericProduct, L2Mixin):
    """FlareList product class.
    This class represents a flare list product in the STIX data processing pipeline.
    It inherits from GenericProduct and L2Mixin, and provides methods to handle flare data.
    It is used to store flare data, and can be enhanced
    with flare positions and SOOP information."""

    LEVEL = "L3"
    TYPE = "flarelist"

    def __init__(self, *, service_type=0, service_subtype=0, ssid, data, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=ssid, data=data, **kwargs)
        self.level = FlareList.LEVEL
        self.type = FlareList.TYPE
        self.service_subtype = 0
        self.service_type = 0
        self._parent = set()

    @property
    def parent(self):
        """Returns the parent(s) of the flare list product.

        Returns
        -------
        list
            A list of parent product names.
        """
        return list(self._parent)

    @parent.setter
    def parent(self, value):
        """Sets the parent of the flare list product.

        Parameters
        ----------
        value : str
            The name of the parent product to be added.
        """
        self._parent.add(value)

    def enhance_from_product(self, in_prod: GenericProduct):
        """Enhances the flare list product by adding more data.

        Parameters
        ----------
        in_prod : GenericProduct
            The input product from which to enhance the flare list product.
        """


class FlarelistSDC(FlareList, FlareSOOPMixin):
    """Base SDC flare-list product (``NAME="sdc"``, ssid 2, level L3).

    Mirrors the operational STIX Data Center flare list and enriches every flare with the
    quicklook peak / quiet-time background counts and fluxes (added by
    `~stixcore.io.FlareListManager.SDCFlareListManager`) and the SOOP campaign (via
    `~stixcore.products.level3.flarelist.FlareSOOPMixin`). It is the first level of the SDC
    chain: `~stixcore.products.level3.flarelist.FlarelistSDC` ->
    `~stixcore.products.level3.flarelist.FlarelistSDCLocation` ->
    `~stixcore.products.level3.flarelist.FlarelistSDCLocationImage`. See
    :doc:`/products/flarelist`.
    """

    PRODUCT_PROCESSING_VERSION = 4
    NAME = "sdc"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=2, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=2, data=data, **kwargs)

        self.name = FlarelistSDC.NAME
        self.ssid = 2

        self._start_datetime = datetime.combine(month, datetime.min.time())
        self._end_datetime = self.data["end_UTC"].max() if len(self.data) > 0 else self._start_datetime

    @property
    def utc_timerange(self):
        return TimeRange(self._start_datetime, self._end_datetime)

    @property
    def scet_timerange(self):
        tr = self.utc_timerange
        logger.warning(
            "scet_timerange will be approximated using Spice. Better to work with utc_timerange property to avoid automatic time conversion"
        )
        start = SCETime.from_string(Spice.instance.datetime_to_scet(tr.start)[2:])
        end = SCETime.from_string(Spice.instance.datetime_to_scet(tr.end)[2:])
        return SCETimeRange(start=start, end=end)

    def split_to_files(self):
        return [self]

    @property
    def dmin(self):
        return (self.data["lc_peak"].sum(axis=1)).min().value if len(self.data) > 0 else np.nan

    @property
    def dmax(self):
        return (self.data["lc_peak"].sum(axis=1)).max().value if len(self.data) > 0 else np.nan

    @property
    def min_exposure(self):
        return self.data["duration"].min().to_value("s") if len(self.data) > 0 else np.nan

    @property
    def max_exposure(self):
        return self.data["duration"].max().to_value("s") if len(self.data) > 0 else np.nan

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 2


class FlarelistSDCLocation(FlarelistSDC, FlarePositionMixin):
    """SDC flare list with a STIX-derived flare location (``NAME="sdcloc"``, ssid 3, level L3).

    Extends `~stixcore.products.level3.flarelist.FlarelistSDC` with the flare position
    (and SOLO position/distance, imaging-quality metric and 1-AU-normalized fluxes) via
    `~stixcore.products.level3.flarelist.FlarePositionMixin`. Only flares above
    ``[Processing] flarelist_sdc_min_count`` peak counts are located. See
    :doc:`/products/flarelist`.
    """

    PRODUCT_PROCESSING_VERSION = 4
    NAME = "sdcloc"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=3, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=3, data=data, month=month, **kwargs)

        self.name = FlarelistSDCLocation.NAME
        self.ssid = 3
        self.location_time_colname = "location_time_UTC"

    def enhance_from_product(self, in_prod: GenericProduct):
        pass

    @classmethod
    def filter_flare_function(cls, col):
        return col["lc_peak"][0].value > CONFIG.getint("Processing", "flarelist_sdc_min_count", fallback=1000)

    @classmethod
    def add_flare_position(cls, data, fido_client: STIXClient, *, month=None):
        super().add_flare_position(
            data,
            fido_client,
            filter_function=cls.filter_flare_function,
            peak_time_colname="peak_UTC",
            start_time_colname="start_UTC",
            end_time_colname="end_UTC",
            keep_all_flares=True,
            month=month,
        )

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 3


class FlarelistSDCLocationImage(FlarelistSDCLocation, FlarePeakPreviewMixin):
    """Located SDC flare list plus peak-preview images (``NAME="sdclocimg"``, ssid 4, level L3).

    Extends `~stixcore.products.level3.flarelist.FlarelistSDCLocation` with per-flare
    peak-preview CLEAN images (`~stixcore.products.level3.flarelistproduct.PeakPreviewImage`),
    reconstructed on the same CPD file used for the flare location, via
    `~stixcore.products.level3.flarelist.FlarePeakPreviewMixin`. See :doc:`/products/flarelist`.
    """

    PRODUCT_PROCESSING_VERSION = 4
    NAME = "sdclocimg"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=4, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=4, data=data, month=month, **kwargs)

        self.name = FlarelistSDCLocationImage.NAME
        self.ssid = 4

    def enhance_from_product(self, in_prod: GenericProduct):
        pass

    @classmethod
    def add_peak_preview(cls, data, energies, parent, fido_client: STIXClient, img_processor, *, month=None):
        super().add_peak_preview(
            data,
            energies,
            parent,
            fido_client,
            img_processor,
            peak_time_colname="peak_UTC",
            start_time_colname="start_UTC",
            end_time_colname="end_UTC",
            anc_ephemeris_path_colname="anc_ephemeris_path",
            cpd_path_colname="cpd_path",
            product_name_suffix=FlarelistSDC.NAME,
            keep_all_flares=False,
            month=month,
        )

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 4


class FlarelistSC(FlareList, FlareSOOPMixin):
    """Flarelist product class for STIXCore flares.

    In L3 product format.
    """

    PRODUCT_PROCESSING_VERSION = 2
    NAME = "sc"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=6, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=6, data=data, **kwargs)

        self.name = FlarelistSC.NAME
        self.ssid = 6

        self._start_datetime = datetime.combine(month, datetime.min.time())
        self._end_datetime = self.data["end_UTC"].max() if len(self.data) > 0 else self._start_datetime

    @property
    def utc_timerange(self):
        return TimeRange(self._start_datetime, self._end_datetime)

    @property
    def scet_timerange(self):
        tr = self.utc_timerange
        logger.warning(
            "scet_timerange will be approximated using Spice. Better to work with utc_timerange property to avoid automatic time conversion"
        )
        start = SCETime.from_string(Spice.instance.datetime_to_scet(tr.start)[2:])
        end = SCETime.from_string(Spice.instance.datetime_to_scet(tr.end)[2:])
        return SCETimeRange(start=start, end=end)

    def split_to_files(self):
        return [self]

    @property
    def dmin(self):
        return (self.data["lc_peak"].sum(axis=1)).min().value if len(self.data) > 0 else np.nan

    @property
    def dmax(self):
        return (self.data["lc_peak"].sum(axis=1)).max().value if len(self.data) > 0 else np.nan

    @property
    def min_exposure(self):
        return self.data["duration"].min().to_value("s") if len(self.data) > 0 else np.nan

    @property
    def max_exposure(self):
        return self.data["duration"].max().to_value("s") if len(self.data) > 0 else np.nan

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 6


class FlarelistSCLocation(FlarelistSC, FlarePositionMixin):
    """Flarelist product class for STIXCore flares.

    In L3 product format.
    """

    PRODUCT_PROCESSING_VERSION = 2
    NAME = "scloc"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=7, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=7, data=data, month=month, **kwargs)

        self.name = FlarelistSCLocation.NAME
        self.ssid = 7
        self.peak_time_colname = "peak_UTC"

    def enhance_from_product(self, in_prod: GenericProduct):
        pass

    @classmethod
    def filter_flare_function(cls, col):
        return col["lc_peak"][0].value > CONFIG.getint("Processing", "flarelist_sdc_min_count", fallback=1000)

    @classmethod
    def add_flare_position(cls, data, fido_client: STIXClient, *, month=None):
        super().add_flare_position(
            data,
            fido_client,
            filter_function=cls.filter_flare_function,
            peak_time_colname="peak_UTC",
            start_time_colname="start_UTC",
            end_time_colname="end_UTC",
            keep_all_flares=False,
            month=month,
        )

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 7


class FlarelistSCLocationImage(FlarelistSCLocation, FlarePeakPreviewMixin):
    """Flarelist product class for StixCore flares.

    In ANC product format.
    """

    PRODUCT_PROCESSING_VERSION = 2
    NAME = "sclocimg"

    def __init__(self, *, service_type=0, service_subtype=0, ssid=8, data, month, **kwargs):
        super().__init__(service_type=0, service_subtype=0, ssid=8, data=data, month=month, **kwargs)

        self.name = FlarelistSCLocationImage.NAME
        self.ssid = 8

    def enhance_from_product(self, in_prod: GenericProduct):
        pass

    @classmethod
    def add_peak_preview(cls, data, energies, parent, fido_client: STIXClient, img_processor, *, month=None):
        super().add_peak_preview(
            data,
            energies,
            parent,
            fido_client,
            img_processor,
            peak_time_colname="peak_UTC",
            start_time_colname="start_UTC",
            end_time_colname="end_UTC",
            anc_ephemeris_path_colname="anc_ephemeris_path",
            cpd_path_colname="cpd_path",
            product_name_suffix=FlarelistSC.NAME,
            keep_all_flares=False,
            month=month,
        )

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return kwargs["level"] == "L3" and service_type == 0 and service_subtype == 0 and ssid == 8


def longest_constant_sequence(state_array):
    """Find the longest sequence where state is constant.
    In case of equal length, prefer the one with the lower state value."""
    if len(state_array) == 0:
        return 0, None, None

    max_length = 0
    max_state = None
    max_start_idx = None
    current_idx = 0

    for state, group in groupby(state_array):
        length = len(list(group))
        # Update if longer, OR if equal length but lower state value
        if length > max_length or (length == max_length and (max_state is None or state < max_state)):
            max_length = length
            max_state = state
            max_start_idx = current_idx
        current_idx += length

    return max_length, max_start_idx, max_state


def calculate_overlap(range1, range2):
    """Calculate the overlap between two TimeRanges.
    Returns the overlap duration and the overlapping TimeRange, or None if no overlap."""

    # Check if they intersect first
    if not range1.intersects(range2):
        return None

    # Calculate intersection boundaries
    overlap_start = max(range1.start, range2.start)
    overlap_end = min(range1.end, range2.end)

    # Create the overlapping TimeRange
    overlap_range = TimeRange(overlap_start, overlap_end)

    return overlap_range
