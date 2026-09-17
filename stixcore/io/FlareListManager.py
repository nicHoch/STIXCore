import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import namedtuple

import numpy as np
import pandas as pd
from stixdcpy.net import Request as stixdcpy_req
from stixpy.calibration.livetime import get_livetime_fraction
from stixpy.product import Product as STIXPYProduct
from sunpy.net import attrs as a

import astropy.units as u
from astropy.table import Column, QTable, vstack
from astropy.time import Time

from stixcore.calibration.elut_manager import ELUTManager
from stixcore.config.config import CONFIG
from stixcore.io.product_processors.fits.processors import CreateUtcColumn
from stixcore.io.RidLutManager import (
    DEFAULT_BKG_EXCLUDE_KEYWORDS,
    DEFAULT_BKG_KEYWORDS,
    RidLutManager,
    search_background_candidates,
)
from stixcore.products.level3.flarelist import FlarelistSC, FlarelistSDC
from stixcore.products.product import Product
from stixcore.util.logging import get_logger
from stixcore.util.singleton import Singleton
from stixcore.util.util import url_to_path

__all__ = [
    "FlareListManager",
    "SDCFlareListManager",
    "SCFlareListManager",
    "compute_ql_count_rate",
    "build_month_timeline",
    "nearest_bin_index",
    "max_rcr_in_window",
    "find_background_file_for_time",
    "BackgroundSelection",
    "background_spectrum_from_cpd",
    "rebin_spectrum_to_ql",
    "rebin_flux_to_ql",
    "collecting_area_cm2",
]

logger = get_logger(__name__)

#: Expected unit of the livetime- and area-corrected flux columns. Computations carry
#: astropy units through and validate against this with ``.to(FLUX_UNIT)`` rather than
#: labelling bare floats.
FLUX_UNIT = u.ct / (u.s * u.keV * u.cm**2)


def compute_ql_count_rate(counts, timedel, triggers, energy_delta, *, n_detectors):
    """Reproduce stixpy's QL count-rate normalization -> ``ct / (s * keV)``.

    Mirrors ``stixpy.timeseries.quicklook`` (lightcurve uses ``n_detectors=16``,
    background uses ``n_detectors=1``). Pure, no I/O.

    Parameters
    ----------
    counts : `~astropy.units.Quantity`
        Raw counts, shape ``(N, 5)`` in ``ct``.
    timedel : `~astropy.units.Quantity`
        Bin durations, shape ``(N,)``.
    triggers : array-like
        Trigger counts, shape ``(N,)`` or ``(N, 1)``.
    energy_delta : `~astropy.units.Quantity`
        Channel widths, shape ``(5,)`` in ``keV``.
    n_detectors : int
        16 for the lightcurve, 1 for the background detector.

    Returns
    -------
    `~astropy.units.Quantity`
        Count rate, shape ``(N, 5)`` in ``ct / (s * keV)``.
    """
    timedel = timedel.to(u.s)
    trig = np.asarray(triggers).reshape(-1)
    live_frac, *_ = get_livetime_fraction(trig / (n_detectors * timedel))
    return counts / ((timedel * live_frac).reshape(-1, 1) * energy_delta)


def build_month_timeline(daily_data_tables):
    """Stack per-day QTables into one time-sorted timeline with unique timestamps.

    ``None``/empty inputs are ignored; an empty list yields an empty ``QTable``.
    After sorting by ``time`` duplicate timestamps (day-boundary overlaps) are
    dropped so the timeline is strictly increasing.
    """
    tables = [t for t in daily_data_tables if t is not None and len(t) > 0]
    if not tables:
        return QTable()
    timeline = vstack(tables, metadata_conflicts="silent")
    timeline.sort("time")
    if len(timeline) > 1:
        keep = np.ones(len(timeline), dtype=bool)
        keep[1:] = np.diff(timeline["time"].jd) > 0
        timeline = timeline[keep]
    return timeline


def nearest_bin_index(times, target, tol):
    """Index of the bin in ``times`` nearest ``target``.

    Returns ``None`` if ``times`` is empty or the nearest gap exceeds ``tol``
    (both ``target`` and ``times`` are `~astropy.time.Time`, ``tol`` a duration).
    """
    if times is None or len(times) == 0:
        return None
    dt = np.abs((times - target).to_value(u.s))
    j = int(np.argmin(dt))
    if dt[j] > tol.to_value(u.s):
        return None
    return j


def max_rcr_in_window(times, rcr, start, end, *, fallback):
    """Highest ``rcr`` for bins with ``start <= time <= end``; ``fallback`` if none."""
    if times is None or len(times) == 0:
        return fallback
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        return fallback
    return int(np.asarray(rcr)[mask].max())


#: 0-based detector indices used for a background spectrum: all 32 science
#: sub-collimators except the Coarse Flare Locator (idx 8) and the Background
#: monitor (idx 9), i.e. the 30 imaging detectors.
_IMAGING_DETECTORS = [d for d in range(32) if d not in (8, 9)]


def _edges_kev(col):
    """Return an energy-edge column as a plain float ndarray in keV."""
    return np.asarray(col.to_value(u.keV) if hasattr(col, "to_value") else col, dtype=float)


_PIXEL_AREA_CACHE = None


def _pixel_area_cm2():
    """12-pixel active-area vector (cm^2) from stixcore's ``stx_subc_params`` config.

    STIX Caliste layout is 8 large pixels then 4 small; pixel sizes are identical
    across detectors so one 12-vector applies to all. Cached after first read.
    """
    global _PIXEL_AREA_CACHE
    if _PIXEL_AREA_CACHE is None:
        import stixcore.config
        from stixcore.config.reader import read_subc_params

        path = Path(stixcore.config.__file__).parent / "data" / "common" / "detector" / "stx_subc_params.csv"
        t = read_subc_params(path)
        large = float(t["L Pixel Xsize"][0]) * float(t["L Pixel Ysize"][0])  # mm^2
        small = float(t["S Pixel Xsize"][0]) * float(t["S Pixel Ysize"][0])  # mm^2
        _PIXEL_AREA_CACHE = np.array([large] * 8 + [small] * 4, dtype=float) / 100.0  # mm^2 -> cm^2
    return _PIXEL_AREA_CACHE


def collecting_area_cm2(product):
    """Geometric collecting area (cm^2) of a detector+pixel-summed product =
    ``detector_mask.sum() * sum(pixel_area[pixel_mask])``. Returns ``None`` when the
    product's control has no detector/pixel mask (e.g. the QL background monitor)."""
    ctrl = getattr(product, "control", None)
    if ctrl is None or "detector_mask" not in ctrl.colnames or "pixel_mask" not in ctrl.colnames:
        return None
    det = np.asarray(ctrl["detector_mask"][0], dtype=bool)
    pix = np.asarray(ctrl["pixel_mask"][0], dtype=bool)
    return float(det.sum()) * float(_pixel_area_cm2()[pix].sum())


def _bkg_monitor_area_cm2():
    """Fixed area (cm^2) of the QL background monitor: one open (grid-less) detector
    over all 12 pixels."""
    return float(_pixel_area_cm2().sum())


def _cpd_present_channel_mask(cpd):
    """Boolean mask (length 32) of which science channels are present in a CPD's
    ``counts`` when only a telemetered subset is stored, or ``None`` if unknown."""
    ctrl = getattr(cpd, "control", None)
    if ctrl is None:
        return None
    if "energy_bin_mask" in ctrl.colnames:
        m = np.asarray(ctrl["energy_bin_mask"][0], dtype=bool).ravel()
        return m if m.size == 32 else None
    if "energy_bin_edge_mask" in ctrl.colnames:
        edges = np.asarray(ctrl["energy_bin_edge_mask"][0], dtype=bool).ravel()
        if edges.size == 33:  # 33 edges -> a channel is present when both its edges are set
            return edges[:-1] & edges[1:]
    return None


def _align_to_energies(spec, nE, energies, cpd):
    """Pad/align a per-channel ``spec`` (length ``nE``) to the energies-table length.

    Preserves an astropy unit on ``spec`` (the NaN padding is created in the same unit,
    so assignment stays unit-checked)."""
    ne_en = len(energies)
    if nE == ne_en:
        return spec
    unit = getattr(spec, "unit", None)
    aligned = np.full(ne_en, np.nan)
    if unit is not None:
        aligned = aligned * unit
    mask = _cpd_present_channel_mask(cpd)
    if mask is not None and mask.size == ne_en and int(mask.sum()) == nE:
        aligned[mask] = spec
    elif nE < ne_en:
        aligned[:nE] = spec
    else:
        logger.warning(f"CPD counts have {nE} channels but energies has {ne_en}; truncating")
        aligned[:] = spec[:ne_en]
    return aligned


def background_spectrum_from_cpd(cpd):
    """Median quiet-period background spectrum (counts and flux) from a CPD product.

    Sums ``counts`` over the 30 imaging detectors (excl. CFL idx 8, BKG monitor idx 9)
    and all pixels, then takes the median over the file's time bins (the whole quiet
    period). ``flux`` is the livetime- and area-normalized rate density
    ``ct s^-1 keV^-1 cm^-2`` (per-bin then median), mirroring stixpy's
    ``create_meta_pixels`` normalization:
    ``flux[t,E] = Σ_dp counts / ((Σ_d livefrac[t,d]·timedel[t]) · (Σ_p area[p]) · dE[E])``.
    Both are aligned to the product's own ``energies`` table (length = telemetered
    channels). Reads ``cpd.data["counts"]`` directly (``get_data`` is not robust on all
    files). Any failure computing the flux yields NaN flux (counts still returned).

    Returns
    -------
    (counts, flux, energies)
    """
    counts = cpd.data["counts"]
    counts_q = counts if hasattr(counts, "unit") else np.asarray(counts, dtype=float) * u.ct  # (nt,32,npix,nE) ct
    nE = counts_q.shape[-1]
    energies = cpd.energies
    csum = counts_q[:, _IMAGING_DETECTORS, :, :].sum(axis=(1, 2))  # (nt, nE) ct over imaging det+pix
    counts_spec = _align_to_energies(np.nanmedian(csum, axis=0), nE, energies, cpd)  # Quantity ct

    flux_spec = np.full(len(energies), np.nan) * FLUX_UNIT
    try:
        timedel = cpd.data["timedel"].to(u.s)  # (nt,) — units carried through the livetime rate
        trig = np.asarray(getattr(cpd.data["triggers"], "value", cpd.data["triggers"]), dtype=float)  # (nt,16)
        from stixpy.calibration.visibility import STIX_INSTRUMENT

        subcol = np.asarray(STIX_INSTRUMENT.subcol_adc_mapping)  # (32,)
        livefrac = get_livetime_fraction(trig[:, subcol] / timedel[:, None])[0]  # (nt, 32), dimensionless
        livefrac = np.asarray(getattr(livefrac, "value", livefrac), dtype=float)
        exp_t = (livefrac[:, _IMAGING_DETECTORS] * timedel[:, None]).sum(axis=1)  # (nt,) s (Σ over dets)
        pmask = _cpd_pixel_mask(cpd)  # which of 12 pixels are telemetered
        pa = _pixel_area_cm2()
        area = (pa[pmask].sum() if pmask is not None else pa[: counts_q.shape[2]].sum()) * u.cm**2  # cm^2
        dE = (energies["e_high"] - energies["e_low"]).to(u.keV)  # (ne_en,) keV
        dE_c = dE if nE == len(energies) else dE[:nE]
        with np.errstate(divide="ignore", invalid="ignore"):
            # astropy derives the unit; .to(FLUX_UNIT) both converts and asserts it is correct
            flux_t = (csum / (exp_t[:, None] * area * dE_c[None, :])).to(FLUX_UNIT)  # (nt, nE)
        flux_spec = _align_to_energies(np.nanmedian(flux_t, axis=0), nE, energies, cpd)
    except Exception as e:
        logger.warning(f"could not compute background flux: {e}")

    return counts_spec, flux_spec, energies


def _cpd_pixel_mask(cpd):
    """Boolean (length 12) of telemetered pixels from the CPD ``pixel_masks`` (row 0),
    or ``None`` when unavailable."""
    data = getattr(cpd, "data", None)
    try:
        if data is not None and "pixel_masks" in data.colnames:
            return np.asarray(data["pixel_masks"][0], dtype=bool).ravel()
    except Exception:
        pass
    return None


def rebin_spectrum_to_ql(counts_32, energies_32, ql_block):
    """Rebin a science *counts* spectrum onto the QL energy bands of ``ql_block``.

    ``ql_block`` carries the target band edges (``e_low``/``e_high``) read from the
    energy table for a given flare (never hardcoded, so a QL binning that changes
    over time is honored). A science channel contributes to a band when its
    ``[e_low, e_high]`` lies within the band; whole channels are summed (the edges
    align). Summing preserves the input unit, so the result carries ``counts_32``'s unit.
    """
    e_low = _edges_kev(energies_32["e_low"])
    e_high = _edges_kev(energies_32["e_high"])
    b_low = _edges_kev(ql_block["e_low"])
    b_high = _edges_kev(ql_block["e_high"])
    unit = getattr(counts_32, "unit", None)
    counts = np.asarray(getattr(counts_32, "value", counts_32), dtype=float)
    out = np.full(len(b_low), np.nan)
    for b in range(len(b_low)):
        chans = (e_low >= b_low[b]) & (e_high <= b_high[b])
        if np.any(chans):
            out[b] = np.nansum(counts[chans])
    return out * unit if unit is not None else out


def rebin_flux_to_ql(flux_32, energies_32, ql_block):
    """Rebin a science *flux* spectrum (per keV) onto the QL bands of ``ql_block``.

    Flux is a density, so bands combine as the dE-weighted mean
    ``flux_band = Σ(flux_ch·dE_ch) / Σ dE_ch`` over each band's channels (edges align).
    The dE weights cancel dimensionally, so the result carries ``flux_32``'s unit.
    """
    e_low = _edges_kev(energies_32["e_low"])
    e_high = _edges_kev(energies_32["e_high"])
    dE = e_high - e_low
    b_low = _edges_kev(ql_block["e_low"])
    b_high = _edges_kev(ql_block["e_high"])
    unit = getattr(flux_32, "unit", None)
    flux = np.asarray(getattr(flux_32, "value", flux_32), dtype=float)
    out = np.full(len(b_low), np.nan)
    for b in range(len(b_low)):
        chans = (e_low >= b_low[b]) & (e_high <= b_high[b])
        denom = np.nansum(dE[chans])
        if np.any(chans) and denom > 0:
            out[b] = np.nansum(flux[chans] * dE[chans]) / denom
    return out * unit if unit is not None else out


def _intern_energy_block(energy, energy_look_up, energies_src):
    """Append ``energies_src`` (channel/e_low/e_high) to the ``energy`` table as a new
    block if its binning is unseen; return ``(energy, index)``. Blocks are hash-deduped
    and get an ``index`` one past the current maximum so they never collide with the
    QL blocks already present."""
    e_sub = QTable()
    e_sub["channel"] = energies_src["channel"]
    e_sub["e_low"] = energies_src["e_low"]
    e_sub["e_high"] = energies_src["e_high"]
    e_hash = frozenset(pd.core.util.hashing.hash_array(e_sub.as_array()))
    if e_hash in energy_look_up:
        return energy, energy_look_up[e_hash]
    idx = (int(np.max(energy["index"])) + 1) if (len(energy) > 0 and "index" in energy.colnames) else 0
    energy_look_up[e_hash] = idx
    e_sub["index"] = Column(idx, description="energy edge table index", dtype=np.int8)
    return vstack([energy, e_sub]), idx


def _ql_block_for(energy, eidx):
    """The rows of the ``energy`` table belonging to block ``eidx`` (a flare's QL
    binning), or ``None`` when the table is empty / has no such block."""
    if energy is None or len(energy) == 0 or "index" not in energy.colnames:
        return None
    sel = np.asarray(energy["index"]) == int(eidx)
    return energy[sel] if np.any(sel) else None


#: Result of :func:`find_background_file_for_time`. ``path`` is a `~pathlib.Path`
#: (or ``None`` when nothing qualifies), ``rid`` the selected BSD request id
#: (``-1`` when none), and ``valid_from``/``valid_to`` the `~astropy.time.Time`
#: interval over which this selection stays valid for a time-ordered caller.
BackgroundSelection = namedtuple("BackgroundSelection", ["path", "rid", "valid_from", "valid_to"])


def _rid_from_filename(path):
    """Parse the BSD request id embedded in a science FITS filename.

    Mirrors ``stixcore.processing.publish`` (the 6th ``_``-separated segment is
    ``<request_id>-<tc>``). Returns ``None`` when the name doesn't carry one.
    """
    parts = Path(path).name.split("_")
    if len(parts) <= 5:
        return None
    try:
        return int(parts[5].replace(".fits", "").split("-")[0])
    except ValueError:
        return None


def _elut_id(time):
    """Identifier of the ELUT active at ``time`` (the resolved ELUT filename), or
    ``None`` when the ELUT index has no unambiguous entry for it. Metadata-only
    lookup via `~stixcore.calibration.elut_manager.ELUTManager` — reads no FITS."""
    try:
        return ELUTManager.instance._find_elut_file(time.to_datetime())
    except Exception as e:
        logger.debug(f"no ELUT resolved for {getattr(time, 'isot', time)}: {e}")
        return None


def _effective_crossover(t0, sa, pa, sb, pb):
    """Earliest ``t >= t0`` (all in JD days) at which candidate ``b``'s effective
    distance ``|t - sb| + pb`` drops to/below ``a``'s ``|t - sa| + pa``; ``None`` if
    it never does. Bounds the nearest-in-time validity interval under the purpose
    penalty (piecewise-linear, breakpoints at the two starts)."""

    def val(x):
        return abs(x - sb) - abs(x - sa) + (pb - pa)

    if val(t0) < 0:
        return t0
    lefts = [t0] + sorted(x for x in (sa, sb) if x > t0)
    for i, left in enumerate(lefts):
        right = lefts[i + 1] if i + 1 < len(lefts) else None
        v = val(left)
        if v < 0:
            return left
        probe = (left + right) / 2 if right is not None else left + 1.0
        slope = ((abs(probe - sb) - abs(probe - sa)) - (abs(left - sb) - abs(left - sa))) / (probe - left)
        if slope < 0:
            tc = left + v / (-slope)
            if right is None or tc <= right:
                return tc
    return None


def find_background_file_for_time(
    time,
    *,
    fido_client,
    rid_lut=None,
    window_past=None,
    window_future=None,
    min_duration=None,
    require_same_elut=None,
    purpose_penalty=None,
    keywords=DEFAULT_BKG_KEYWORDS,
    exclude_keywords=None,
    exclude_flare_comment=None,
):
    """Find the best quiet-time background CPD file applicable at ``time``.

    Strategy (see :func:`stixcore.io.RidLutManager.search_background_candidates`):
    rank background requests from the RID LUT **nearest-in-time first** (closest
    request start, past or future, within the separate ``window_past`` /
    ``window_future`` bounds). For each candidate resolve the real ``sci_xray_cpd``
    L1 file(s) via FIDO, keep those whose filename carries the candidate rid, and
    accept the first one that passes the "good background" checks (requested long
    enough, attenuator out over the whole interval).

    Three optional stricter filters exist but default **off** (they move common
    cases, not just edge cases): a ``purpose == "Background"`` preference
    (``purpose_penalty`` > 0), dropping "elevated" backgrounds (``exclude_keywords``),
    and dropping requests whose comment references a specific flare id
    (``exclude_flare_comment``). See the ``[Processing]`` config keys.

    Alongside the winning file a validity interval ``[valid_from, valid_to]`` is
    returned so a time-ordered caller (monthly flare processing) can reuse the
    result for every later time inside the interval. ``valid_to`` is the earliest
    time at which another candidate's effective distance would overtake the chosen
    one, or the chosen start plus ``window_past`` when there is none — capped at
    ``time + window_future`` so newly-reachable candidates are re-scanned. A
    "no file found" result is cached until the next candidate could appear.

    Parameters
    ----------
    time : `~astropy.time.Time` or str
        The query time (e.g. a flare peak).
    fido_client : `~stixpy.net.client.STIXClient`
        Client used for the ``sci_xray_cpd`` L1 search.
    rid_lut : `~astropy.table.Table`, optional
        The RID LUT; defaults to ``RidLutManager.instance.rid_lut``.
    window_past, window_future : `~astropy.units.Quantity`, optional
        Search windows; default to the ``[Processing]`` config keys
        ``flarelist_bkg_window_past_days`` / ``flarelist_bkg_window_future_days``.
    min_duration : `~astropy.units.Quantity`, optional
        Minimum requested integration time; defaults to
        ``flarelist_bkg_min_duration_s``.
    require_same_elut : bool, optional
        If True, a candidate is only accepted when the ELUT active at its request
        start (per `~stixcore.calibration.elut_manager.ELUTManager`) matches the
        one active at ``time`` — i.e. the background was taken under the same
        on-board ELUT configuration as the flare. Defaults to the ``[Processing]``
        config key ``flarelist_bkg_require_same_elut``.
    purpose_penalty : `~astropy.units.Quantity`, optional
        Distance penalty for non-``Background``-purpose candidates; defaults to the
        ``[Processing]`` config key ``flarelist_bkg_purpose_penalty_days``.
    keywords, exclude_keywords : tuple of str, optional
        Positive / negative background keywords for the candidate search.
    exclude_flare_comment : bool, optional
        Drop candidates whose comment references a specific flare id.

    Returns
    -------
    BackgroundSelection
    """
    if window_past is None:
        window_past = CONFIG.getfloat("Processing", "flarelist_bkg_window_past_days", fallback=30.0) * u.day
    if window_future is None:
        window_future = CONFIG.getfloat("Processing", "flarelist_bkg_window_future_days", fallback=7.0) * u.day
    if min_duration is None:
        min_duration = CONFIG.getfloat("Processing", "flarelist_bkg_min_duration_s", fallback=1200.0) * u.s
    if require_same_elut is None:
        require_same_elut = CONFIG.getboolean("Processing", "flarelist_bkg_require_same_elut", fallback=True)
    if purpose_penalty is None:
        purpose_penalty = CONFIG.getfloat("Processing", "flarelist_bkg_purpose_penalty_days", fallback=0.0) * u.day
    if exclude_keywords is None:
        exclude_keywords = (
            DEFAULT_BKG_EXCLUDE_KEYWORDS
            if CONFIG.getboolean("Processing", "flarelist_bkg_exclude_elevated", fallback=False)
            else ()
        )
    if exclude_flare_comment is None:
        exclude_flare_comment = CONFIG.getboolean("Processing", "flarelist_bkg_exclude_flare_comment", fallback=False)

    t = time if isinstance(time, Time) else Time(time)
    if rid_lut is None:
        rid_lut = RidLutManager.instance.rid_lut

    # ELUT active at the flare time; only enforced when it can be resolved
    flare_elut = _elut_id(t) if require_same_elut else None

    candidates = search_background_candidates(
        rid_lut,
        t,
        window_past=window_past,
        window_future=window_future,
        keywords=keywords,
        exclude_keywords=exclude_keywords,
        exclude_flare_comment=exclude_flare_comment,
        purpose_penalty=purpose_penalty,
    )

    t_jd = t.to_value("jd")
    penalty_days = purpose_penalty.to_value(u.day)

    def _valid_to(chosen):
        # the selection holds until another candidate's *effective* distance overtakes
        # the chosen one's (accounts for the purpose penalty), or the chosen leaves the
        # past window if none does. Capped at time + window_future so newly-reachable
        # candidates get re-scanned.
        pa = 0.0 if chosen.is_background else penalty_days
        sa = chosen.start.to_value("jd")
        edge = sa + window_past.to_value(u.day)
        for j in candidates:
            if j.rid == chosen.rid:
                continue
            pb = 0.0 if j.is_background else penalty_days
            tc = _effective_crossover(t_jd, sa, pa, j.start.to_value("jd"), pb)
            if tc is not None:
                edge = min(edge, tc)
        return min(Time(edge, format="jd", scale="utc"), t + window_future)

    for cand in candidates:
        if (cand.end - cand.start) < min_duration:
            logger.debug(f"bkg candidate rid {cand.rid}: requested duration below {min_duration}; skipping")
            continue
        if flare_elut is not None and _elut_id(cand.start) != flare_elut:
            logger.debug(f"bkg candidate rid {cand.rid}: different ELUT configuration; skipping")
            continue
        try:
            res = fido_client.search(
                a.Time(cand.start, cand.end),
                a.Instrument.stix,
                a.stix.DataProduct.sci_xray_cpd,
                a.Level("L1"),
            )
        except Exception as e:
            logger.warning(f"bkg candidate rid {cand.rid}: CPD search failed: {e}")
            continue
        if len(res) == 0:
            continue
        res.filter_for_latest_version()
        url_to_path(res)
        if "path" not in res.columns:
            continue
        for path in res["path"]:
            if path is None:
                continue
            if _rid_from_filename(path) != cand.rid:
                continue
            try:
                p = STIXPYProduct(path)
            except Exception as e:
                logger.warning(f"bkg candidate rid {cand.rid}: could not load {path}: {e}")
                continue
            rcr = p.data["rcr"] if "rcr" in p.data.colnames else None
            if rcr is None or np.any(np.asarray(rcr) != 0):
                logger.debug(f"bkg candidate rid {cand.rid}: attenuator in (rcr != 0); skipping {path}")
                continue
            valid_to = _valid_to(cand)
            logger.info(f"selected background file {path} (rid {cand.rid}, {cand.side}) for {t.isot}")
            return BackgroundSelection(path=Path(str(path)), rid=cand.rid, valid_from=t, valid_to=valid_to)

    later = [c.start for c in candidates if c.start > t]
    valid_to = min(later) if later else t + window_future
    logger.info(f"no usable background file found for {t.isot}")
    return BackgroundSelection(path=None, rid=-1, valid_from=t, valid_to=valid_to)


class FlareListManager:
    """Base class for the flare-list source managers.

    Holds the source flare list and the product class it feeds, and provides the shared
    enrichment machinery that turns raw flare definitions into an enriched flare list:
    `~stixcore.io.FlareListManager.FlareListManager._build_ql_month_timeline`,
    `~stixcore.io.FlareListManager.FlareListManager.add_lc_bkg_columns` (quicklook peak /
    background counts and fluxes) and
    `~stixcore.io.FlareListManager.FlareListManager.add_background_file_column` (quiet-time
    background spectrum). The active subclass is
    `~stixcore.io.FlareListManager.SDCFlareListManager`, which mirrors the operational STIX
    Data Center flare list. See :doc:`/products/flarelist`.
    """

    @property
    def flarelist(self):
        return self._flarelist

    @flarelist.setter
    def flarelist(self, value):
        self._flarelist = value

    @property
    def flarelistname(self):
        return type(self).__name__

    @property
    def productCls(self):
        return self._product_cls

    def _build_ql_month_timeline(self, *, start, end, fido_client, data_product, n_detectors, track_energy):
        """Search + load all L1 QL files of ``data_product`` for ``[start, end)`` and
        stack them into one time-sorted timeline with a per-bin count-rate column.

        Each daily file is opened exactly once (used for both the energy-table
        lookup and the counts), so the whole month costs one open per day.

        Returns
        -------
        (timeline, energy, date_to_eidx)
            ``timeline`` : QTable with columns ``time``, ``counts`` (raw ``ct``,
            ``(N, 5)``), ``counts_rate`` (``(N, 5)``) and, for the lightcurve,
            ``rcr``. ``energy`` and ``date_to_eidx`` are only populated when
            ``track_energy`` is True.
        """
        energy = QTable()
        energy_look_up = {}
        date_to_eidx = {}
        daily_tables = []

        try:
            res = fido_client.search(a.Time(start, end), a.Instrument.stix, data_product, a.Level("L1"))
            if len(res) > 0:
                res.filter_for_latest_version()
                url_to_path(res)
        except Exception as e:
            logger.error(f"error searching L1 QL {data_product} files for month {start}: {e}")
            return QTable(), energy, date_to_eidx

        if len(res) == 0 or "path" not in res.columns:
            return QTable(), energy, date_to_eidx

        for path in res["path"]:
            if path is None:
                continue
            try:
                p = STIXPYProduct(path)
            except Exception as e:
                logger.warning(f"could not load QL product {path}: {e}")
                continue

            energies = getattr(p, "_energies", None)
            if energies is None:
                logger.warning(f"QL product {path} has no energies table; skipping")
                continue

            counts = p.data["counts"]
            if counts.shape[1] != 5:
                logger.warning(f"QL product {path} has {counts.shape[1]} channels (expected 5); skipping")
                continue

            energy_delta = energies["e_high"] - energies["e_low"]
            rate = compute_ql_count_rate(
                counts, p.data["timedel"], p.data["triggers"], energy_delta, n_detectors=n_detectors
            )
            # area-normalize to a flux (ct/s/keV/cm^2): use the product's detector/pixel
            # masks, falling back to the fixed open-detector area for the QL background monitor
            area = collecting_area_cm2(p)
            if area is None:
                area = _bkg_monitor_area_cm2()
            rate = rate / (area * u.cm**2)

            daily = QTable()
            daily["time"] = p.data["time"]
            daily["counts"] = counts
            daily["counts_rate"] = rate
            if "rcr" in p.data.colnames:
                daily["rcr"] = np.asarray(p.data["rcr"]).astype(np.int16)
            daily_tables.append(daily)

            if track_energy:
                e_sub = QTable()
                e_sub["channel"] = energies["channel"]
                e_sub["e_low"] = energies["e_low"]
                e_sub["e_high"] = energies["e_high"]
                e_hash = frozenset(pd.core.util.hashing.hash_array(e_sub.as_array()))
                if e_hash not in energy_look_up:
                    e_idx = len(energy_look_up)
                    energy_look_up[e_hash] = e_idx
                    e_sub["index"] = Column(e_idx, description="energy edge table index", dtype=np.int8)
                    energy = vstack([energy, e_sub])
                    if e_idx > 0:
                        logger.warning(f"multiple energy ql-lc tables found for month {start}")
                eidx = energy_look_up[e_hash]
                for d in {t.to_datetime().date() for t in p.data["time"]}:
                    date_to_eidx.setdefault(d, eidx)
            logger.info(f"loaded QL product {path} with {len(p.data)} bins and {len(energies)} energy channels")

        return build_month_timeline(daily_tables), energy, date_to_eidx

    def add_lc_bkg_columns(self, data, *, start, end, fido_client):
        """Populate LC/BKG peak counts + flux, RCR and ``att_in`` on ``data`` from the
        real L1 QL lightcurve + background products, and return the energy QTable.

        ``data`` must already carry ``flare_id`` and the astropy ``Time`` columns
        ``start_UTC`` / ``end_UTC`` / ``peak_UTC``. Columns are added in place:
        ``lc_peak``, ``lc_peak_flux``, ``lc_bkg_peak``, ``lc_bkg_peak_flux``,
        ``rcr_at_peak``, ``rcr_max``, ``att_in``, ``energy_index``. ``*_flux`` are
        livetime- and area-normalized (ct/s/keV/cm2).
        """
        n = len(data)
        tol = CONFIG.getfloat("Processing", "flarelist_peak_max_dist_s", fallback=60.0) * u.s

        lc_timeline, energy, date_to_eidx = self._build_ql_month_timeline(
            start=start,
            end=end,
            fido_client=fido_client,
            data_product=a.stix.DataProduct.ql_lightcurve,
            n_detectors=16,
            track_energy=True,
        )
        bkg_timeline, _, _ = self._build_ql_month_timeline(
            start=start,
            end=end,
            fido_client=fido_client,
            data_product=a.stix.DataProduct.ql_background,
            n_detectors=1,
            track_energy=False,
        )

        # unit-carrying accumulators: assigning a Quantity slice below is unit-checked, so
        # a wrong-unit timeline value would raise rather than be silently stored
        # raw counts kept as unsigned int32 (u.Quantity floats by default; explicit dtype keeps
        # it uint32 -> FITS 'J' + unsigned TZERO, 4 B/elem exact). No-match rows stay 0, never NaN.
        lc_peak = u.Quantity(np.zeros((n, 5), dtype=np.uint32), u.ct, dtype=np.uint32)
        lc_peak_flux = np.zeros((n, 5), dtype=np.float32) * FLUX_UNIT
        lc_bkg_peak = u.Quantity(np.zeros((n, 5), dtype=np.uint32), u.ct, dtype=np.uint32)
        lc_bkg_peak_flux = np.zeros((n, 5), dtype=np.float32) * FLUX_UNIT
        rcr_at_peak = np.full(n, -1, dtype=np.int8)
        rcr_max = np.full(n, -1, dtype=np.int8)
        energy_index = np.zeros(n, dtype=np.int8)

        lc_has = len(lc_timeline) > 0
        bkg_has = len(bkg_timeline) > 0
        if not lc_has:
            logger.warning(f"No L1 QL lightcurve data found for month {start}")
        if not bkg_has:
            logger.warning(f"No L1 QL background data found for month {start}")

        for i, row in enumerate(data):
            peak = row["peak_UTC"]
            fid = row["flare_id"]
            if lc_has:
                j = nearest_bin_index(lc_timeline["time"], peak, tol)
                if j is not None:
                    lc_peak[i] = lc_timeline["counts"][j]  # Quantity ct -> unit-checked assignment
                    lc_peak_flux[i] = lc_timeline["counts_rate"][j].to(FLUX_UNIT)  # assert flux unit
                    rcr_at_peak[i] = int(lc_timeline["rcr"][j])
                    rcr_max[i] = max_rcr_in_window(
                        lc_timeline["time"],
                        lc_timeline["rcr"],
                        row["start_UTC"],
                        row["end_UTC"],
                        fallback=int(rcr_at_peak[i]),
                    )
                    energy_index[i] = date_to_eidx.get(peak.to_datetime().date(), 0)
                else:
                    logger.warning(f"flare {fid}: no LC bin within {tol} of peak {peak.isot}")
            if bkg_has:
                k = nearest_bin_index(bkg_timeline["time"], peak, tol)
                if k is not None:
                    lc_bkg_peak[i] = bkg_timeline["counts"][k]  # Quantity ct -> unit-checked assignment
                    lc_bkg_peak_flux[i] = bkg_timeline["counts_rate"][k].to(FLUX_UNIT)  # assert flux unit
                else:
                    logger.warning(f"flare {fid}: no BKG bin within {tol} of peak {peak.isot}")

        # Drop orphan energy-binning blocks: a QL product with a different binning can add
        # rows to `energy` that no flare references (date_to_eidx keeps the first binning
        # seen per date). Keep only referenced blocks and renumber energy_index contiguously.
        if len(energy) > 0 and "index" in energy.colnames:
            used = sorted({int(x) for x in energy_index})
            remap = {old: new for new, old in enumerate(used)}
            energy = energy[[k for k, e in enumerate(energy["index"]) if int(e) in remap]]
            energy["index"] = np.array([remap[int(x)] for x in energy["index"]], dtype=energy["index"].dtype)
            energy_index = np.array([remap[int(x)] for x in energy_index], dtype=energy_index.dtype)

        data["lc_peak"] = Column(
            lc_peak,  # already a Quantity in ct
            description="raw counts at the L1 QL lightcurve bin nearest the flare peak (5 energy channels)",
            dtype=np.uint32,
        )
        data["lc_peak_flux"] = Column(
            lc_peak_flux,  # already a Quantity in ct/s/keV/cm2
            description="livetime- and area-corrected flux at the peak lightcurve bin (ct/s/keV/cm2, 5 energy channels)",
        )
        data["lc_bkg_peak"] = Column(
            lc_bkg_peak,  # already a Quantity in ct
            description="raw counts from the STIX background detector at the L1 QL background bin nearest the flare peak (5 energy channels) — unmodulated by imaging subcollimators and not affected by the attenuator",
            dtype=np.uint32,
        )
        data["lc_bkg_peak_flux"] = Column(
            lc_bkg_peak_flux,  # already a Quantity in ct/s/keV/cm2
            description="livetime- and area-corrected STIX background detector flux at the peak bin (ct/s/keV/cm2, 5 energy channels)",
        )
        data["rcr_at_peak"] = Column(
            rcr_at_peak, description="rate control regime at the peak bin (>0 attenuator in)", dtype=np.int8
        )
        data["rcr_max"] = Column(
            rcr_max, description="max rate control regime over the flare start..end window", dtype=np.int8
        )
        data["att_in"] = Column(rcr_max > 0, description="was attenuator in during flare (rcr_max > 0)")
        data["energy_index"] = Column(energy_index, description="energy band index", dtype=np.int8)

        # these CSV-derived placeholders (added in get_data) are superseded here; drop if present
        for _col in ("bkg_baseline", "bkg_quiet_period"):
            if _col in data.colnames:
                del data[_col]

        return energy

    def add_background_file_column(self, data, *, energy, fido_client):
        """Add background-file and quiet-period background-spectrum columns per flare.

        Writes ``bkg_file`` / ``bkg_rid`` (the best quiet-time background CPD file for
        each flare peak) and, extracted from that file, the median quiet-period
        background spectrum: raw counts in the native science channels (``bkg_spec``) and
        rebinned to the flare's QL bands (``bkg_spec_ql``), plus the livetime- and
        area-normalized flux (ct/s/keV/cm2) in native channels (``bkg_spec_flux``) and
        rebinned to the QL bands (``bkg_spec_flux_ql``); ``bkg_energy_index`` points at the
        native binning appended to the ``energy`` table.

        ``data`` rows are assumed peak-time ascending, so the per-time background
        search (:func:`find_background_file_for_time`) is cached with its validity
        interval and only re-run when a flare peak leaves that period. The extracted
        spectrum is likewise cached per background file id, so each CPD is opened only
        once no matter how many flares share it. Requires the ``energy`` table (from
        :meth:`add_lc_bkg_columns`) so the QL rebin follows the actual QL binning;
        returns the (possibly extended, orphan-pruned) ``energy`` table.
        """
        n = len(data)
        bkg_files = [""] * n
        bkg_rids = np.full(n, -1, dtype=np.int64)
        spec_rows = [None] * n  # native-binning counts spectrum per flare (Quantity ct, length varies)
        flux_rows = [None] * n  # native-binning flux spectrum per flare (Quantity, ct/s/keV/cm2)
        bkg_spec_ql = np.full((n, 5), np.nan) * u.ct  # native counts rebinned to QL bands
        bkg_spec_flux_ql = np.full((n, 5), np.nan) * FLUX_UNIT  # unit-checked on assignment below
        bkg_energy_index = np.full(n, -1, dtype=np.int16)
        bkg_solo_sun_distance = np.full(n, np.nan) * u.km  # SOLO-Sun distance at the bkg CPD's time

        primer = ""
        baseurl = getattr(fido_client, "baseurl", None)
        datapath = getattr(fido_client, "datapath", None)
        if baseurl is not None and datapath is not None:
            primer = baseurl.replace(datapath, "")
            primer = primer[7:] if primer.startswith("file://") else primer

        energy_look_up_32 = {}  # native binning hash -> index in `energy`
        spectrum_cache = {}  # rid -> (counts, flux, energies, eidx, dsun_km)
        rebin_cache = {}  # (rid, energy_index) -> flux_ql (5,)
        selection = None
        searches = 0
        opens = 0
        for i, row in enumerate(data):
            peak = row["peak_UTC"]
            if selection is None or not (selection.valid_from <= peak <= selection.valid_to):
                selection = find_background_file_for_time(peak, fido_client=fido_client)
                searches += 1
            if selection.path is None:
                continue
            bkg_files[i] = str(selection.path).replace(primer, "")
            bkg_rids[i] = selection.rid

            if selection.rid not in spectrum_cache:
                try:
                    cpd = STIXPYProduct(selection.path)
                    c32, f32, e32 = background_spectrum_from_cpd(cpd)
                    energy, eidx32 = _intern_energy_block(energy, energy_look_up_32, e32)
                    meta = getattr(cpd, "meta", None) or {}
                    dsun = float(meta["DSUN_OBS"]) * u.m if "DSUN_OBS" in meta else np.nan * u.m
                    spectrum_cache[selection.rid] = (c32, f32, e32, eidx32, dsun.to(u.km))
                    opens += 1
                except Exception as e:
                    logger.warning(f"could not extract background spectrum from {selection.path}: {e}")
                    spectrum_cache[selection.rid] = None
            cached = spectrum_cache[selection.rid]
            if cached is None:
                continue
            c32, f32, e32, eidx32, dsun = cached
            spec_rows[i] = c32
            flux_rows[i] = f32
            bkg_energy_index[i] = eidx32
            bkg_solo_sun_distance[i] = dsun

            eidx_ql = int(data["energy_index"][i]) if "energy_index" in data.colnames else -1
            ql_block = _ql_block_for(energy, eidx_ql)
            if ql_block is not None:
                key = (selection.rid, eidx_ql)
                if key not in rebin_cache:
                    rebin_cache[key] = (
                        rebin_spectrum_to_ql(c32, e32, ql_block),
                        rebin_flux_to_ql(f32, e32, ql_block),
                    )
                bkg_spec_ql[i], bkg_spec_flux_ql[i] = rebin_cache[key]

        # Drop orphan energy blocks (keep those referenced by either the QL energy_index
        # or the background bkg_energy_index) and renumber both index columns contiguously.
        if len(energy) > 0 and "index" in energy.colnames:
            ref_ql = {int(x) for x in data["energy_index"]} if "energy_index" in data.colnames else set()
            ref_bkg = {int(x) for x in bkg_energy_index if x >= 0}
            used = sorted(ref_ql | ref_bkg)
            remap = {old: new for new, old in enumerate(used)}
            energy = energy[[k for k, e in enumerate(energy["index"]) if int(e) in remap]]
            energy["index"] = np.array([remap[int(x)] for x in energy["index"]], dtype=energy["index"].dtype)
            if "energy_index" in data.colnames:
                data["energy_index"] = np.array(
                    [remap.get(int(x), 0) for x in data["energy_index"]], dtype=data["energy_index"].dtype
                )
            bkg_energy_index = np.array([remap[int(x)] if x >= 0 else -1 for x in bkg_energy_index], dtype=np.int16)

        # native-binning spectra can differ in length across files (telemetered channel
        # count varies); build rectangular Quantity columns at the widest, padding with NaN.
        # Assigning the Quantity rows into the Quantity arrays is unit-checked.
        width = max((len(r) for r in spec_rows if r is not None), default=32)
        bkg_spec = np.full((n, width), np.nan) * u.ct
        bkg_spec_flux = np.full((n, width), np.nan) * FLUX_UNIT
        for i in range(n):
            if spec_rows[i] is not None:
                bkg_spec[i, : len(spec_rows[i])] = spec_rows[i]
            if flux_rows[i] is not None:
                bkg_spec_flux[i, : len(flux_rows[i])] = flux_rows[i]

        data["bkg_file"] = Column(bkg_files, description="path to the quiet-time background CPD file")
        data["bkg_rid"] = Column(
            bkg_rids, description="BSD request id of the selected background file (-1 if none)", dtype=np.int64
        )
        data["bkg_spec"] = Column(
            bkg_spec,  # already a Quantity in ct
            description="median quiet-period background counts per science energy channel (30 imaging detectors)",
        )
        data["bkg_spec_ql"] = Column(
            bkg_spec_ql,  # already a Quantity in ct
            description="median quiet-period background counts rebinned to the QL lightcurve energy bands",
        )
        data["bkg_spec_flux"] = Column(
            bkg_spec_flux,  # already a Quantity in ct/s/keV/cm2
            description="median quiet-period background flux per science channel (ct/s/keV/cm2, 30 imaging detectors)",
        )
        data["bkg_spec_flux_ql"] = Column(
            bkg_spec_flux_ql,  # already a Quantity in ct/s/keV/cm2
            description="median quiet-period background flux rebinned to the QL lightcurve energy bands (ct/s/keV/cm2)",
        )
        data["bkg_energy_index"] = Column(
            bkg_energy_index,
            description="energy table index of the native background binning (-1 if no background file)",
            dtype=np.int16,
        )
        data["bkg_solo_sun_distance"] = Column(
            bkg_solo_sun_distance,  # already a Quantity in km
            description="SOLO-Sun distance at the background CPD's observation time (from DSUN_OBS; NaN if none)",
        )
        logger.info(f"background file search ran {searches}x, opened {opens} CPD files for {n} flares")
        return energy


class SCFlareListManager(FlareListManager, metaclass=Singleton):
    """Manages a local copy of the flarelist provided by STIXCore or runs the flare detection

    TODO
    """

    def __init__(self, file, fido_client, update=False):
        """Creates the manager by pointing to the flarelist file (csv) and setting the update
           strategy.

        Parameters
        ----------
        file : Path
            points to the csv file
        update : bool, optional
            Update strategy: is the flarelist file updated via API?, by default False
        """
        self.file = file
        self.update = update
        self._product_cls = FlarelistSC
        self.fido_client = fido_client
        self._flarelist = self.read_flarelist()

    def __str__(self) -> str:
        return f"{self.flarelistname}: file: {self.file} update: {self.update} size: {len(self.flarelist)}"

    def update_list(self):
        """Updates the flarelist file via api request.

        Will create a new file if not available or do a incremental update otherwise,
        using the last entry time stamp.
        """
        self.flarelist = SDCFlareListManager.read_flarelist(self.file, update=self.update)

    def read_flarelist(self):
        """Reads or creates the LUT of all STIXCore flares.

        On creation or update a flare detection is run on the STIXCore data
        to get the information and persists as a LUT locally.

        Parameters
        ----------
        file : Path
            path the to LUT file.
        update : bool, optional
            should the LUT be updated at start up?, by default False

        Returns
        -------
        Table
            the table of flares.
        """
        if self.update or not self.file.exists():
            last_date = datetime(2020, 1, 1, 0, 0, 0)
            # only run flare detection for time older than 7 days
            today = datetime.now() - timedelta(days=7)
            flare_df_lists = []
            if self.file.exists():
                old_list = pd.read_csv(self.file, keep_default_na=True, na_values=["None"])
                mds = old_list["start_UTC"].max()
                try:
                    last_date = datetime.strptime(mds, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    last_date = datetime.strptime(mds, "%Y-%m-%dT%H:%M:%S.%f")
                flare_df_lists = [old_list]
            last_date = last_date.replace(hour=0, minute=0, second=0, microsecond=0)
            if not self.file.parent.exists():
                logger.info(f"path not found: {self.file.parent} creating dir")
                self.file.parent.mkdir(parents=True, exist_ok=True)

            try:
                while last_date < today:
                    # run flare detection for batches of 7 days
                    # 2 days overlap to ensure no flares are missed
                    start = last_date - timedelta(days=2)
                    end = last_date + timedelta(days=7)

                    ql_lc_files = self.fido_client.search(
                        a.Time(start, end), a.Instrument.stix, a.stix.DataProduct.ql_lightcurve
                    )
                    try:
                        if len(ql_lc_files) > 0:
                            ql_lc_files.filter_for_latest_version()
                            url_to_path(ql_lc_files)
                    except Exception as e:
                        logger.error(f"Error filtering for latest version of lightcurve files: {e}")
                        ql_lc_files = []

                    ql_bg_files = self.fido_client.search(
                        a.Time(start, end), a.Instrument.stix, a.stix.DataProduct.ql_background
                    )
                    if len(ql_bg_files) > 0:
                        ql_bg_files.filter_for_latest_version()
                        url_to_path(ql_bg_files)
                    logger.info(
                        f"flare detection chunk: {start.isoformat()}/{end.isoformat()} "
                        f"with {len(ql_lc_files)} lightcurve files and "
                        f"{len(ql_bg_files)} background files"
                    )
                    # TODO re-enable flare-list creation
                    # flares = stixpy.detect_flares(start, end,
                    #                               ql_lc_files=ql_lc_files,
                    #                               ql_bg_files=ql_bg_files)

                    flares = []
                    if len(flares) > 0:
                        flare_df_lists.append(pd.DataFrame(flares))
                        logger.info(f"found {len(flares)} flares")
                    last_date += timedelta(days=7)

            except Exception:
                logger.error("FLARE DETECTION ERROR", exc_info=True)

            full_flare_list = flare_df_lists
            # full_flare_list = pd.concat(flare_df_lists)

            # full_flare_list.drop_duplicates(inplace=True)
            # full_flare_list.sort_values(by="peak_UTC", inplace=True)
            # full_flare_list.reset_index(inplace=True, drop=True)
            logger.info(f"write total {len(full_flare_list)} flares to local storage")
            # full_flare_list.to_csv(self.file, index_label=False)
        else:
            logger.info(f"read flare list from {self.file}")
            full_flare_list = pd.read_csv(self.file, keep_default_na=True, na_values=["None"])

        return full_flare_list

    @staticmethod
    def filter_flare_function(col):
        return col["lc_peak"][0].value > CONFIG.getint("Processing", "flarelist_sc_min_count", fallback=1000)

    def get_data(self, *, start, end, fido_client):
        month_data = self.flarelist[
            (self.flarelist["start_UTC"] >= start.isoformat()) & (self.flarelist["start_UTC"] < end.isoformat())
        ]

        if len(month_data) == 0:
            return None, None, None

        mt = QTable(month_data.to_numpy(), names=month_data.columns)
        data = QTable()
        control = QTable()
        energy = QTable()

        data["flare_id"] = Column(
            mt["flare_id"].astype(int), description=f"unique flare id for flarelist {self.flarelistname}"
        )
        CreateUtcColumn(
            data,
            [Time(d, format="isot", scale="utc") for d in mt["start_UTC"]],
            "start_UTC",
            description="start time of flare",
        )

        data["duration"] = Column(mt["duration"].astype(float) * u.s, description="duration of flare")
        data["end_UTC"] = CreateUtcColumn(description="end time of flare")
        data["end_UTC"] = [Time(d, format="isot", scale="utc") for d in mt["end_UTC"]]
        data["peak_UTC"] = CreateUtcColumn(description="flare peak time")
        data["peak_UTC"] = [Time(d, format="isot", scale="utc") for d in mt["peak_UTC"]]
        data["att_in"] = Column(mt["att_in"].astype(bool), description="was attenuator in during flare")
        data["GOES_class"] = Column(
            mt["GOES_class"].astype(str),
            description="GOES class of the GOES XRS data at time of flare"
            " - not derived from STIX data.  Do not use when "
            "flare isn't visible to Earth",
        )
        data["goes_min_class_est"] = Column(
            mt["goes_estimated_min_class"].astype(str), description="min GOES class estimate derived from STIX data"
        )
        data["goes_max_class_est"] = Column(
            mt["goes_estimated_max_class"].astype(str), description="max GOES class estimate derived from STIX data"
        )
        data["goes_mean_class_est"] = Column(
            mt["goes_estimated_mean_class"].astype(str), description="mean GOES class estimate derived from STIX data"
        )

        data["GOES_flux"] = Column(
            mt["GOES_flux"].astype(float) * u.W / u.m**2,
            description="GOES flux of the GOES XRS data at time of flare"
            "- not derived from STIX data. Do not use when the "
            "flare isn't visible to Earth",
        )
        data["goes_min_flux_est"] = Column(
            (10 ** mt["goes_estimated_min_flux"].astype(float)) * u.W / u.m**2,
            description="min GOES flux estimate derived from STIX data",
        )
        data["goes_max_flux_est"] = Column(
            (10 ** mt["goes_estimated_max_flux"].astype(float)) * u.W / u.m**2,
            description="max GOES flux estimate derived from STIX data",
        )
        data["goes_mean_flux_est"] = Column(
            (10 ** mt["goes_estimated_mean_flux"].astype(float)) * u.W / u.m**2,
            description="mean GOES flux estimate derived from STIX data",
        )

        # data['cfl_x'] = Column(mt['CFL_X_arcsec'].astype(float) * u.arcsec,
        #                        description="coarse flare location in x direction provided by"
        #                                    "onboard algorithm. (0,0) represents disk center")
        # data['cfl_y'] = Column(mt['CFL_Y_arcsec'].astype(float) * u.arcsec,
        #                        description="coarse flare location in y direction provided by"
        #                                    "onboard algorithm. (0,0) represents disk center")

        data["lc_peak"] = Column(
            (
                np.vstack(
                    (
                        mt["LC0_PEAK_COUNTS_4S"].value,
                        mt["LC1_PEAK_COUNTS_4S"].value,
                        mt["LC2_PEAK_COUNTS_4S"].value,
                        mt["LC3_PEAK_COUNTS_4S"].value,
                        mt["LC4_PEAK_COUNTS_4S"].value,
                    )
                ).T
                * u.ct
            ).astype(int),
            description="counts in 4s peak window from quicklook lightcurve",
            dtype=np.int64,
        )

        data["lc_bkg_peak"] = Column(
            (
                np.vstack(
                    (
                        mt["LC0_BKG_COUNTS_4S"].value,
                        mt["LC1_BKG_COUNTS_4S"].value,
                        mt["LC2_BKG_COUNTS_4S"].value,
                        mt["LC3_BKG_COUNTS_4S"].value,
                        mt["LC4_BKG_COUNTS_4S"].value,
                    )
                ).T
                * u.ct
            ).astype(int),
            description="background counts in 4s peak windowfrom quicklook lightcurve",
            dtype=np.int64,
        )

        data["energy_index"] = Column(0, description="energy band index", dtype=np.int8)

        data.add_index("flare_id")

        # add energy axis for the lightcurve peak time data for each flare
        # the energy bins are taken from the daily ql-lightcurve products
        # as the definition of the lc energy chanel's are will change only very seldom
        # the ql-lightcurve products assume a constant definition for an entire day.
        # So we do the lookup also just grouped by peak day in order to save file lookups

        energy_look_up = {}
        data["peak_day"] = [d.datetime.day for d in data["peak_UTC"]]
        data_by_day = data.group_by("peak_day")

        for day, flares in zip(data_by_day.groups.keys, data_by_day.groups):
            time = flares["peak_UTC"][0]
            lc_data = fido_client.search(a.Time(time, time), a.Instrument.stix, a.stix.DataProduct.ql_lightcurve)
            lc_data.filter_for_latest_version()
            url_to_path(lc_data)

            if len(lc_data) == 0:
                logger.warning(f"No lightcurve data found for flare at time {time}")
                continue
            lc = Product(lc_data["path"][0])

            energy_table_hash = frozenset(pd.core.util.hashing.hash_array(lc.energies.as_array()))

            # add the energy table to the energy table list if not already
            #  present and define a new index number
            if energy_table_hash not in energy_look_up:
                e_idx = len(energy_look_up.keys())
                energy_look_up[energy_table_hash] = e_idx
                lc.energies["index"] = Column(e_idx, description="energy edge table index", dtype=np.int8)
                energy = vstack([energy, lc.energies])
                if e_idx > 0:
                    logger.warning(f"multiple energy ql-lc tables found for month {start}")

            # add the energy index to the flare data to all flares of the same day
            # https://docs.astropy.org/en/latest/table/modify_table.html#caveats
            replace = data.loc[flares["flare_id"]]
            replace["energy_index"] = energy_look_up[energy_table_hash]
            data.loc[flares["flare_id"]] = replace

        del data["peak_day"]

        return data, control, energy


class SDCFlareListManager(FlareListManager, metaclass=Singleton):
    """Manages a local copy of the operational flarelist provided by stix data datacenter

    TODO
    """

    def __init__(self, file, update=False):
        """Creates the manager by pointing to the flarelist file (csv) and setting the update
           strategy.

        Parameters
        ----------
        file : Path
            points to the csv file
        update : bool, optional
            Update strategy: is the flarelist file updated via API?, by default False
        """
        self.file = file
        self.update = update
        self._flarelist = SDCFlareListManager.read_flarelist(self.file, self.update)
        self._product_cls = FlarelistSDC

    def __str__(self) -> str:
        return f"{self.flarelistname}: file: {self.file} update: {self.update} size: {len(self.flarelist)}"

    def update_list(self):
        """Updates the flarelist file via api request.

        Will create a new file if not available or do a incremental update otherwise,
        using the last entry time stamp.
        """
        self.flarelist = SDCFlareListManager.read_flarelist(self.file, update=self.update)

    @classmethod
    def read_flarelist(cls, file, update=False):
        """Read the local flare-list CSV mirror, optionally refreshing it from the STIX Data Center.

        When ``update`` is set (or the file does not yet exist) the operational flare list is
        fetched from the STIX Data Center via ``stixdcpy.fetch_flare_list`` in ~monthly chunks
        (the API is batched by month and throttled, so the loop sleeps between chunks) from
        2020-01-01 to now; an incremental update re-fetches the last ~60 days. The chunks are
        concatenated, de-duplicated, sorted by ``peak_UTC`` and cached back to ``file``.

        Parameters
        ----------
        file : Path
            Path to the local flare-list CSV mirror.
        update : bool, optional
            Refresh from the STIX Data Center before reading, by default False.

        Returns
        -------
        `~pandas.DataFrame`
            The full flare list.
        """
        if update or not file.exists():
            # the api is limited to batch sizes of a month. in order to get the full table we have
            # to ready each month after the start of STIX
            last_date = datetime(2020, 1, 1, 0, 0, 0)
            today = datetime.now()  # - timedelta(days=60)
            flare_df_lists = []
            if file.exists():
                old_list = pd.read_csv(file, keep_default_na=True, na_values=["None"])
                mds = old_list["start_UTC"].max()
                try:
                    last_date = datetime.strptime(mds, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    last_date = datetime.strptime(mds, "%Y-%m-%dT%H:%M:%S.%f")
                flare_df_lists = [old_list]
            last_date -= timedelta(days=60)
            if not file.parent.exists():
                logger.info(f"path not found to flare list file dir: {file.parent} creating dir")
                file.parent.mkdir(parents=True, exist_ok=True)

            try:
                while last_date < today:
                    last_date_1m = last_date + timedelta(days=30)
                    logger.info(f"download flare list chunk: {last_date.isoformat()}/{last_date_1m.isoformat()}")
                    flares = stixdcpy_req.fetch_flare_list(last_date.isoformat(), last_date_1m.isoformat())
                    last_date = last_date_1m
                    if len(flares) > 0:
                        flare_df_lists.append(pd.DataFrame(flares))
                        logger.info(f"found {len(flares)} flares")
                    # the stix datacenter API is throttled to 2 calls per second
                    time.sleep(0.5)
            except Exception:
                logger.error("FLARELIST API ERROR", exc_info=True)

            full_flare_list = pd.concat(flare_df_lists)

            full_flare_list.drop_duplicates(inplace=True)
            full_flare_list.sort_values(by="peak_UTC", inplace=True)
            full_flare_list.reset_index(inplace=True, drop=True)
            logger.info(f"write total {len(full_flare_list)} flares to local storage")
            full_flare_list.to_csv(file, index_label=False)
        else:
            logger.info(f"read flare list from {file}")
            full_flare_list = pd.read_csv(file, keep_default_na=True, na_values=["None"])

        return full_flare_list

    @staticmethod
    def filter_flare_function(col):
        return col["lc_peak"][0].value > CONFIG.getint("Processing", "flarelist_sdc_min_count", fallback=1000)

    def get_data(self, *, start, end, fido_client):
        """Build the enriched SDC flare list for the ``[start, end)`` month.

        Slices the requested month from the local CSV mirror and builds the base columns
        (``flare_id``, the UTC time columns, GOES class/flux and the source quiet-period
        background ``bkg_baseline`` / ``bkg_quiet_period``). It then enriches each flare with
        the quicklook peak/background counts and fluxes
        (`~stixcore.io.FlareListManager.FlareListManager.add_lc_bkg_columns`) and the quiet-time
        background spectrum
        (`~stixcore.io.FlareListManager.FlareListManager.add_background_file_column`). The
        source CSV's own at-peak lightcurve counts and attenuator flag are unreliable and are
        replaced by these STIX-derived values. See :doc:`/products/flarelist`.

        Parameters
        ----------
        start, end : `~datetime.datetime`
            Half-open month boundaries; flares with ``start <= start_UTC < end`` are kept.
        fido_client : `~stixpy.net.client.STIXClient`
            Client used to resolve the quicklook and CPD files during enrichment.

        Returns
        -------
        tuple
            ``(data, control, energy)`` QTables, or ``(None, None, None)`` if the month is empty.
        """
        month_data = self.flarelist[
            (self.flarelist["start_UTC"] >= start.isoformat()) & (self.flarelist["start_UTC"] < end.isoformat())
        ]

        if len(month_data) == 0:
            return None, None, None

        mt = QTable(month_data.to_numpy(), names=month_data.columns)
        data = QTable()
        control = QTable()
        energy = QTable()

        data["flare_id"] = Column(
            mt["flare_id"].astype(int), description=f"unique flare id for flarelist {self.flarelistname}"
        )

        CreateUtcColumn(
            data,
            [Time(d, format="isot", scale="utc") for d in mt["start_UTC"]],
            "start_UTC",
            description="start time of flare",
        )
        data["duration"] = Column(mt["duration"].astype(float) * u.s, description="duration of flare")

        CreateUtcColumn(
            data,
            [Time(d, format="isot", scale="utc") for d in mt["end_UTC"]],
            "end_UTC",
            description="end time of flare",
        )
        CreateUtcColumn(
            data,
            [Time(d, format="isot", scale="utc") for d in mt["peak_UTC"]],
            "peak_UTC",
            description="flare peak time",
        )

        data["GOES_class"] = Column(
            mt["GOES_class"].astype(str),
            description="GOES class of the GOES XRS data at time of flare"
            " - not derived from STIX data.  Do not use when "
            "flare isn't visible to Earth",
        )
        data["goes_min_class_est"] = Column(
            mt["goes_estimated_min_class"].astype(str), description="min GOES class estimate derived from STIX data"
        )
        data["goes_max_class_est"] = Column(
            mt["goes_estimated_max_class"].astype(str), description="max GOES class estimate derived from STIX data"
        )
        data["goes_mean_class_est"] = Column(
            mt["goes_estimated_mean_class"].astype(str), description="mean GOES class estimate derived from STIX data"
        )

        data["GOES_flux"] = Column(
            mt["GOES_flux"].astype(float) * u.W / u.m**2,
            description="GOES flux of the GOES XRS data at time of flare"
            "- not derived from STIX data. Do not use when the "
            "flare isn't visible to Earth",
        )
        data["goes_min_flux_est"] = Column(
            (10 ** mt["goes_estimated_min_flux"].astype(float)) * u.W / u.m**2,
            description="min GOES flux estimate derived from STIX data",
        )
        data["goes_max_flux_est"] = Column(
            (10 ** mt["goes_estimated_max_flux"].astype(float)) * u.W / u.m**2,
            description="max GOES flux estimate derived from STIX data",
        )
        data["goes_mean_flux_est"] = Column(
            (10 ** mt["goes_estimated_mean_flux"].astype(float)) * u.W / u.m**2,
            description="mean GOES flux estimate derived from STIX data",
        )

        # data['cfl_x'] = Column(mt['CFL_X_arcsec'].astype(float) * u.arcsec,
        #                        description="coarse flare location in x direction provided by"
        #                                    "onboard algorithm. (0,0) represents disk center")
        # data['cfl_y'] = Column(mt['CFL_Y_arcsec'].astype(float) * u.arcsec,
        #                        description="coarse flare location in y direction provided by"
        #                                    "onboard algorithm. (0,0) represents disk center")

        # background estimates carried over from the source flare list (distinct from the
        # QL-derived at-peak background added below)
        data["bkg_baseline"] = Column(
            mt["LC0_BKG"].astype(float) * u.ct,
            description="median value of the fitted baseline",
        )
        data["bkg_quiet_period"] = Column(
            (
                np.vstack(
                    (
                        mt["LC0_BKG_COUNTS_4S"].value,
                        mt["LC1_BKG_COUNTS_4S"].value,
                        mt["LC2_BKG_COUNTS_4S"].value,
                        mt["LC3_BKG_COUNTS_4S"].value,
                        mt["LC4_BKG_COUNTS_4S"].value,
                    )
                ).T
                * u.ct
            ).astype(np.int64),
            description="background counts per QL energy channel, median value for the most recent quiet period",
            dtype=np.int64,
        )

        # LC/BKG peak counts+rates, RCR, att_in and energy_index are derived from the
        # real L1 QL lightcurve + background products (the source CSV values are
        # unreliable) using one monthly timeline per product built once.
        energy = self.add_lc_bkg_columns(data, start=start, end=end, fido_client=fido_client)

        # select the best quiet-time background data file for each flare peak and extract
        # its median quiet-period background spectrum (adds the 32-ch binning to `energy`)
        energy = self.add_background_file_column(data, energy=energy, fido_client=fido_client)

        data.add_index("flare_id")

        return data, control, energy


if "pytest" in sys.modules:
    # only set the global in test scenario
    from stixcore.data.test import test_data

    SDCFlareListManager.instance = SDCFlareListManager(test_data.rid_lut.RID_LUT, update=False)
