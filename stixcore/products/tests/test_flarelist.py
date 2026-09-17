from datetime import date

import numpy as np
import pytest
from sunpy.coordinates import HeliographicStonyhurst
from sunpy.time import TimeRange

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time

from stixcore.io.product_processors.fits.processors import FitsL3Processor
from stixcore.products.level3.flarelist import (
    FlarelistSDCLocation,
    FlarePositionResult,
    _empty_flare_position,
    add_distance_normalized_flux,
    calculate_overlap,
    cpd_timedel_range,
    longest_constant_sequence,
)
from stixcore.products.product import Product

N = 10


@pytest.fixture
def flare_data():
    peak_times = Time("2022-01-01T12:00:00") + np.arange(N) * 600 * u.s

    lon = np.linspace(0, 30, N)
    lat = np.linspace(-5, 5, N)
    # mark same rows fully NaN so the entire SkyCoord row is invalid
    lon[2] = lat[2] = np.nan
    lon[7] = lat[7] = np.nan

    hgs_coords = SkyCoord(
        lon=lon * u.deg,
        lat=lat * u.deg,
        radius=np.ones(N) * 1.0 * u.AU,
        frame=HeliographicStonyhurst(obstime=peak_times),
    )

    data = QTable()
    data["peak_UTC"] = peak_times
    data["location_time_UTC"] = peak_times
    data["start_UTC"] = peak_times - 60 * u.s
    data["end_UTC"] = peak_times + 60 * u.s
    data["duration"] = np.ones(N) * 120 * u.s
    data["lc_peak"] = np.ones((N, 5)) * u.ct / u.s
    data["location_hgs"] = hgs_coords

    return data


@pytest.fixture
def written_fits(flare_data, tmp_path):
    prod = FlarelistSDCLocation(
        data=flare_data,
        month=date(2022, 1, 1),
        control=QTable(),
    )

    # minimal header bypasses the Spice-dependent header generation chain
    header = fits.Header()
    header["LEVEL"] = "L3"
    header["STYPE"] = 0
    header["SSTYPE"] = 0
    header["SSID"] = 3
    header["DATE-BEG"] = "2022-01-01T00:00:00"
    prod.fits_header = header
    prod.energy = None
    prod._additional_header_keywords = []

    writer = FitsL3Processor(tmp_path)
    written = writer.write_fits(prod)
    assert len(written) == 1

    return prod, written[0]


def test_flarelist_sdcloc_location_roundtrip(written_fits):
    prod, fits_path = written_fits
    orig_hgs_lon = prod.data["location_hgs"].lon.copy()
    orig_hgs_lat = prod.data["location_hgs"].lat.copy()

    # read back via Product factory — calls on_deserialize internally
    recovered = Product(fits_path)

    assert isinstance(recovered, FlarelistSDCLocation)
    assert_quantity_allclose(recovered.data["location_hgs"].lon, orig_hgs_lon, atol=1e-6 * u.deg, equal_nan=True)
    assert_quantity_allclose(recovered.data["location_hgs"].lat, orig_hgs_lat, atol=1e-6 * u.deg, equal_nan=True)


def test_flarelist_sdcloc_fits_stores_icrs(written_fits):
    prod, fits_path = written_fits
    orig_hgs_lon = prod.data["location_hgs"].lon.copy()
    orig_hgs_lat = prod.data["location_hgs"].lat.copy()

    # read the DATA extension directly — no on_deserialize, raw FITS content
    raw = QTable.read(fits_path, hdu="DATA", astropy_native=True)

    assert "location_hgs" not in raw.colnames, "HGS column should not be stored in FITS"
    assert "location_icrs" in raw.colnames, "ICRS column should be present in FITS"

    # manually transform ICRS back to HGS and compare with original
    obstime = Time(raw["location_time_UTC"])
    hgs = raw["location_icrs"].transform_to(HeliographicStonyhurst(obstime=obstime))
    assert_quantity_allclose(hgs.lon, orig_hgs_lon, atol=1e-6 * u.deg, equal_nan=True)
    assert_quantity_allclose(hgs.lat, orig_hgs_lat, atol=1e-6 * u.deg, equal_nan=True)


# --- longest_constant_sequence ---


def test_lcs_empty():
    length, start, state = longest_constant_sequence([])
    assert length == 0
    assert start is None
    assert state is None


def test_lcs_single_element():
    length, start, state = longest_constant_sequence([5])
    assert length == 1
    assert start == 0
    assert state == 5


def test_lcs_all_same():
    length, start, state = longest_constant_sequence([3, 3, 3, 3])
    assert length == 4
    assert start == 0
    assert state == 3


def test_lcs_clear_winner():
    length, start, state = longest_constant_sequence([1, 2, 2, 2, 3, 3])
    assert length == 3
    assert start == 1
    assert state == 2


def test_lcs_tie_prefers_lower_state():
    # two runs of length 2: state=1 at index 0, state=3 at index 2
    length, start, state = longest_constant_sequence([1, 1, 3, 3])
    assert length == 2
    assert start == 0
    assert state == 1


def test_lcs_numpy_array():
    arr = np.array([0, 0, 1, 1, 1, 0])
    length, start, state = longest_constant_sequence(arr)
    assert length == 3
    assert start == 2
    assert state == 1


# --- calculate_overlap ---


def test_overlap_no_intersection():
    r1 = TimeRange("2024-01-01T00:00:00", "2024-01-01T01:00:00")
    r2 = TimeRange("2024-01-01T02:00:00", "2024-01-01T03:00:00")
    assert calculate_overlap(r1, r2) is None


def test_overlap_partial():
    r1 = TimeRange("2024-01-01T00:00:00", "2024-01-01T02:00:00")
    r2 = TimeRange("2024-01-01T01:00:00", "2024-01-01T03:00:00")
    result = calculate_overlap(r1, r2)
    assert result is not None
    assert result.start == Time("2024-01-01T01:00:00")
    assert result.end == Time("2024-01-01T02:00:00")


def test_overlap_contained():
    r1 = TimeRange("2024-01-01T00:00:00", "2024-01-01T04:00:00")
    r2 = TimeRange("2024-01-01T01:00:00", "2024-01-01T03:00:00")
    result = calculate_overlap(r1, r2)
    assert result is not None
    assert result.start == Time("2024-01-01T01:00:00")
    assert result.end == Time("2024-01-01T03:00:00")


def test_overlap_identical():
    r1 = TimeRange("2024-01-01T00:00:00", "2024-01-01T01:00:00")
    result = calculate_overlap(r1, r1)
    assert result is not None
    assert result.start == r1.start
    assert result.end == r1.end


# --- add_distance_normalized_flux ---

FLUX_UNIT = u.ct / (u.s * u.keV * u.cm**2)


def test_add_distance_normalized_flux():
    data = QTable()
    # flare-time distance (for LC) and background-time distance (for bkg) differ per row
    data["solo_sun_distance"] = [0.5, 1.0, np.nan] * u.AU
    data["bkg_solo_sun_distance"] = [0.8, 1.0, 0.5] * u.AU
    data["lc_peak_flux"] = np.array([[100.0, 10, 1, 5, 2]] * 3) * FLUX_UNIT
    data["bkg_spec_flux"] = np.array([[4.0, 3, 2, 1, 0.5]] * 3) * FLUX_UNIT
    # lc_bkg_peak_flux / bkg_spec_flux_ql intentionally absent -> must be skipped, no error

    add_distance_normalized_flux(data)

    # new columns added, originals unchanged
    assert "lc_peak_flux_at_1au" in data.colnames
    assert "bkg_spec_flux_at_1au" in data.colnames
    assert "lc_bkg_peak_flux_at_1au" not in data.colnames  # source column was absent
    assert np.allclose(data["lc_peak_flux"].to_value(FLUX_UNIT), [[100, 10, 1, 5, 2]] * 3)

    # LC uses solo_sun_distance: 0.5 AU -> x0.25, 1 AU -> x1, NaN -> NaN
    lc = data["lc_peak_flux_at_1au"].to_value(FLUX_UNIT)
    assert np.allclose(lc[0], np.array([100, 10, 1, 5, 2]) * 0.25)
    assert np.allclose(lc[1], [100, 10, 1, 5, 2])
    assert np.all(np.isnan(lc[2]))
    # bkg uses bkg_solo_sun_distance: 0.8 AU -> x0.64, 0.5 AU (row 2) -> x0.25
    bk = data["bkg_spec_flux_at_1au"].to_value(FLUX_UNIT)
    assert np.allclose(bk[0], np.array([4, 3, 2, 1, 0.5]) * 0.64)
    assert np.allclose(bk[2], np.array([4, 3, 2, 1, 0.5]) * 0.25)
    # unit preserved
    assert data["lc_peak_flux_at_1au"].unit.is_equivalent(FLUX_UNIT)


# --- cpd_timedel_range ---


def test_cpd_timedel_range():
    # four bins at t = 0,10,20,30 s with widths 4,4,10,20 ds (0.4,0.4,1.0,2.0 s)
    times = np.array([0, 10, 20, 30]) * u.s
    timedels = np.array([4, 4, 10, 20]) * u.ds

    # window fully inside the file -> bins 1 and 2 overlap [5, 25] s
    ds_min, ds_max = cpd_timedel_range(times, timedels, 5 * u.s, 25 * u.s)
    assert ds_min.unit == u.ds
    assert ds_max.unit == u.ds
    assert ds_min == 4 * u.ds
    assert ds_max == 10 * u.ds

    # window extends beyond the file (not fully covered) -> all bins overlap
    ds_min, ds_max = cpd_timedel_range(times, timedels, -100 * u.s, 100 * u.s)
    assert ds_min == 4 * u.ds
    assert ds_max == 20 * u.ds

    # no overlap -> (NaN ds, NaN ds)
    ds_min, ds_max = cpd_timedel_range(times, timedels, 100 * u.s, 200 * u.s)
    assert ds_min.unit == u.ds
    assert ds_max.unit == u.ds
    assert np.isnan(ds_min.value)
    assert np.isnan(ds_max.value)


# --- _empty_flare_position ---


def test_empty_flare_position_defaults():
    peak = Time("2022-01-01T12:00:00")
    r = _empty_flare_position(peak)

    assert isinstance(r, FlarePositionResult)
    assert r.status is False
    assert r.message == ""
    assert r.anc_path == ""
    assert r.cpd_path == ""
    assert r.rcr_at_peak == 0
    assert r.solo_time == peak  # per-row placeholder passed through
    # NaN geometry
    for f in (r.flare_x, r.flare_y, r.flare_z, r.solo_x, r.solo_y, r.solo_z):
        assert f.unit == u.km
        assert np.isnan(f.value)
    assert np.isnan(r.sidelobe)
    # new ds columns default to NaN deciseconds
    assert r.min_exposure.unit == u.ds
    assert np.isnan(r.min_exposure.value)
    assert r.max_exposure.unit == u.ds
    assert np.isnan(r.max_exposure.value)

    # overrides applied
    r2 = _empty_flare_position(peak, message="no CPD data found", anc_path="/some/anc.fits")
    assert r2.message == "no CPD data found"
    assert r2.anc_path == "/some/anc.fits"
    assert r2.status is False  # untouched
