from pathlib import Path

import numpy as np
import pytest

import astropy.units as u
from astropy.table import QTable, Table
from astropy.time import Time

import stixcore.io.FlareListManager as flm_mod
from stixcore.io.FlareListManager import (
    BackgroundSelection,
    FlareListManager,
    background_spectrum_from_cpd,
    build_month_timeline,
    collecting_area_cm2,
    compute_ql_count_rate,
    find_background_file_for_time,
    max_rcr_in_window,
    nearest_bin_index,
    rebin_flux_to_ql,
    rebin_spectrum_to_ql,
)
from stixcore.io.RidLutManager import RidLutManager, search_background_candidates

RATE_UNIT = u.ct / (u.s * u.keV)
FLUX_UNIT = u.ct / (u.s * u.keV * u.cm**2)


# --- helpers to build synthetic QL data ---------------------------------------


def _energies():
    e = QTable()
    e["channel"] = np.arange(5, dtype=np.uint8)
    e["e_low"] = [4, 10, 15, 25, 50] * u.keV
    e["e_high"] = [10, 15, 25, 50, 84] * u.keV
    return e


def _ql_data(t0, n, *, rcr=None, counts_value=100, with_rcr=True):
    """Build a synthetic QL product ``.data`` QTable on a 4 s grid."""
    times = Time(t0) + np.arange(n) * 4 * u.s
    data = QTable()
    data["time"] = times
    data["timedel"] = np.full(n, 4.0) * u.s
    data["triggers"] = np.zeros(n)  # zero triggers -> live_frac == 1 (deterministic)
    data["counts"] = (np.full((n, 5), counts_value)).astype(int) * u.ct
    if with_rcr:
        data["rcr"] = np.zeros(n, dtype=np.ubyte) if rcr is None else np.asarray(rcr, dtype=np.ubyte)
    return data


class FakeProduct:
    def __init__(self, data, energies):
        self.data = data
        self._energies = energies


class FakeResponse:
    """Minimal stand-in for a StixQueryResponse carrying a ``path`` column."""

    def __init__(self, paths):
        self._paths = list(paths)

    def __len__(self):
        return len(self._paths)

    @property
    def columns(self):
        return ["path"] if self._paths else []

    def filter_for_latest_version(self):
        pass

    def __getitem__(self, key):
        assert key == "path"
        return self._paths


class FakeFido:
    def __init__(self, lc_paths, bkg_paths):
        self.lc_paths = lc_paths
        self.bkg_paths = bkg_paths

    def search(self, time, instrument, data_product, level):
        name = getattr(data_product, "value", str(data_product))
        if "background" in name:
            return FakeResponse(self.bkg_paths)
        return FakeResponse(self.lc_paths)


# --- compute_ql_count_rate ----------------------------------------------------


def test_count_rate_units_and_shape():
    data = _ql_data("2024-06-15T12:00:00", 3)
    ed = _energies()["e_high"] - _energies()["e_low"]
    rate = compute_ql_count_rate(data["counts"], data["timedel"], data["triggers"], ed, n_detectors=16)
    assert rate.shape == (3, 5)
    assert rate.unit.is_equivalent(RATE_UNIT)


def test_count_rate_zero_triggers_is_deterministic():
    # zero triggers => live_frac == 1 => rate == counts / (timedel * energy_delta),
    # independent of n_detectors, so LC and BKG agree exactly.
    data = _ql_data("2024-06-15T12:00:00", 2, counts_value=200)
    ed = _energies()["e_high"] - _energies()["e_low"]
    lc = compute_ql_count_rate(data["counts"], data["timedel"], data["triggers"], ed, n_detectors=16)
    bkg = compute_ql_count_rate(data["counts"], data["timedel"], data["triggers"], ed, n_detectors=1)
    expected = data["counts"] / ((data["timedel"]).reshape(-1, 1) * ed)
    assert u.allclose(lc, expected.to(RATE_UNIT))
    assert u.allclose(lc, bkg)


def test_count_rate_more_detectors_gives_smaller_rate_when_triggers_nonzero():
    # nonzero triggers: larger n_detectors -> lower trigger rate -> higher live_frac
    # -> larger denominator -> smaller count rate.
    data = _ql_data("2024-06-15T12:00:00", 1)
    data["triggers"] = np.array([5000.0])
    ed = _energies()["e_high"] - _energies()["e_low"]
    lc = compute_ql_count_rate(data["counts"], data["timedel"], data["triggers"], ed, n_detectors=16)
    bkg = compute_ql_count_rate(data["counts"], data["timedel"], data["triggers"], ed, n_detectors=1)
    assert np.all(lc.to_value(RATE_UNIT) < bkg.to_value(RATE_UNIT))


# --- build_month_timeline -----------------------------------------------------


def test_build_timeline_sorts_and_dedups_overlap():
    d1 = _ql_data("2024-06-15T12:00:00", 3)  # 12:00:00, :04, :08
    d2 = _ql_data("2024-06-15T12:00:08", 3)  # :08 (dup), :12, :16
    timeline = build_month_timeline([d2, d1])  # pass out of order
    t = timeline["time"]
    assert len(timeline) == 5  # 6 bins minus 1 duplicate at :08
    assert np.all(np.diff(t.jd) > 0)  # strictly increasing


def test_build_timeline_empty():
    assert len(build_month_timeline([])) == 0
    assert len(build_month_timeline([None, QTable()])) == 0


# --- nearest_bin_index --------------------------------------------------------


def test_nearest_bin_exact():
    times = Time("2024-06-15T12:00:00") + np.arange(5) * 4 * u.s
    assert nearest_bin_index(times, times[2], 2 * u.s) == 2


def test_nearest_bin_within_tol():
    times = Time("2024-06-15T12:00:00") + np.arange(5) * 4 * u.s
    target = times[3] + 1 * u.s
    assert nearest_bin_index(times, target, 2 * u.s) == 3


def test_nearest_bin_beyond_tol_returns_none():
    times = Time("2024-06-15T12:00:00") + np.arange(5) * 4 * u.s
    target = times[-1] + 1 * u.h
    assert nearest_bin_index(times, target, 60 * u.s) is None


def test_nearest_bin_empty_returns_none():
    assert nearest_bin_index(Time([], format="isot"), Time("2024-06-15T12:00:00"), 60 * u.s) is None


# --- max_rcr_in_window --------------------------------------------------------


def test_max_rcr_over_window():
    times = Time("2024-06-15T12:00:00") + np.arange(6) * 4 * u.s
    rcr = np.array([0, 0, 1, 2, 1, 0])
    assert max_rcr_in_window(times, rcr, times[1], times[4], fallback=-1) == 2


def test_max_rcr_peak_only_window():
    times = Time("2024-06-15T12:00:00") + np.arange(6) * 4 * u.s
    rcr = np.array([0, 0, 1, 2, 1, 0])
    assert max_rcr_in_window(times, rcr, times[0], times[0], fallback=-1) == 0


def test_max_rcr_empty_window_returns_fallback():
    times = Time("2024-06-15T12:00:00") + np.arange(6) * 4 * u.s
    rcr = np.array([0, 0, 1, 2, 1, 0])
    before = Time("2024-06-15T11:00:00")
    assert max_rcr_in_window(times, rcr, before, before, fallback=7) == 7


# --- add_lc_bkg_columns (integration, monkeypatched) --------------------------


@pytest.fixture
def flare_data():
    # bin grid starts 2024-06-15T12:00:00, 4 s cadence; bins 5-7 attenuated (rcr=1)
    peaks = Time(
        [
            "2024-06-15T12:00:20",  # bin 5 (attenuated), window covers attenuated bins
            "2024-06-15T12:00:00",  # bin 0, no attenuation
            "2024-06-15T13:00:00",  # far away -> beyond tolerance
        ]
    )
    data = QTable()
    data["flare_id"] = [1, 2, 3]
    data["start_UTC"] = peaks - 8 * u.s
    data["end_UTC"] = peaks + 8 * u.s
    data["peak_UTC"] = peaks
    return data


def _patch_fido(monkeypatch):
    n = 20
    rcr = np.zeros(n, dtype=np.ubyte)
    rcr[5:8] = 1
    lc = FakeProduct(_ql_data("2024-06-15T12:00:00", n, rcr=rcr), _energies())
    bkg = FakeProduct(_ql_data("2024-06-15T12:00:00", n, with_rcr=False), _energies())

    products = {"lc": lc, "bkg": bkg}
    monkeypatch.setattr("stixcore.io.FlareListManager.STIXPYProduct", lambda path: products[path])
    return FakeFido(lc_paths=["lc"], bkg_paths=["bkg"])


def test_add_lc_bkg_columns(flare_data, monkeypatch):
    fido = _patch_fido(monkeypatch)
    from datetime import date

    energy = FlareListManager().add_lc_bkg_columns(
        flare_data, start=date(2024, 6, 1), end=date(2024, 7, 1), fido_client=fido
    )

    # shapes / units (QTable stores unit-bearing columns as float64 Quantity,
    # same as the pre-existing lc_peak column; values stay integral counts)
    assert flare_data["lc_peak"].shape == (3, 5)
    assert flare_data["lc_peak"].unit == u.ct
    assert np.all(flare_data["lc_peak"].value == np.round(flare_data["lc_peak"].value))
    assert flare_data["lc_peak_flux"].unit.is_equivalent(FLUX_UNIT)
    assert flare_data["lc_bkg_peak_flux"].unit.is_equivalent(FLUX_UNIT)
    assert flare_data["lc_bkg_peak"].shape == (3, 5)
    assert flare_data["lc_bkg_peak"].unit == u.ct
    assert flare_data["att_in"].dtype == bool
    assert flare_data["energy_index"].dtype == np.int8

    # rcr semantics
    assert np.all(flare_data["rcr_max"] >= flare_data["rcr_at_peak"])
    assert list(flare_data["att_in"]) == [True, False, False]
    assert flare_data["rcr_max"][0] == 1  # window covers attenuated bins
    assert flare_data["rcr_at_peak"][0] == 1
    assert flare_data["rcr_at_peak"][1] == 0

    # far flare -> beyond tolerance -> zero filled, rcr -1
    assert np.all(flare_data["lc_peak"][2].to_value(u.ct) == 0)
    assert flare_data["rcr_at_peak"][2] == -1
    assert flare_data["rcr_max"][2] == -1

    # counts pulled from the real timeline for in-range flares
    assert np.all(flare_data["lc_peak"][0].to_value(u.ct) == 100)
    assert np.all(flare_data["lc_bkg_peak"][0].to_value(u.ct) == 100)

    # returned energy table schema
    assert set(energy.colnames) == {"channel", "e_low", "e_high", "index"}
    assert len(energy) == 5


def test_add_lc_bkg_columns_no_files(flare_data, monkeypatch):
    from datetime import date

    fido = FakeFido(lc_paths=[], bkg_paths=[])
    energy = FlareListManager().add_lc_bkg_columns(
        flare_data, start=date(2024, 6, 1), end=date(2024, 7, 1), fido_client=fido
    )

    assert np.all(flare_data["lc_peak"].to_value(u.ct) == 0)
    assert np.all(flare_data["lc_bkg_peak"].to_value(u.ct) == 0)
    assert np.all(flare_data["rcr_at_peak"] == -1)
    assert np.all(flare_data["rcr_max"] == -1)
    assert not np.any(flare_data["att_in"])
    assert len(energy) == 0


def _energies_alt():
    """A second, different QL energy binning (distinct edges -> distinct hash)."""
    e = QTable()
    e["channel"] = np.arange(5, dtype=np.uint8)
    e["e_low"] = [4, 11, 16, 26, 51] * u.keV
    e["e_high"] = [11, 16, 26, 51, 85] * u.keV
    return e


def test_energy_table_prunes_orphan_binning(flare_data, monkeypatch):
    from datetime import date

    # two LC products for the month with DIFFERENT binnings, both on the same date;
    # date_to_eidx keeps the first (primary) binning, so the alternate block is an orphan.
    n = 20
    rcr = np.zeros(n, dtype=np.ubyte)
    rcr[5:8] = 1
    lc = FakeProduct(_ql_data("2024-06-15T12:00:00", n, rcr=rcr), _energies())
    lc_alt = FakeProduct(_ql_data("2024-06-15T12:00:00", n, rcr=rcr), _energies_alt())
    bkg = FakeProduct(_ql_data("2024-06-15T12:00:00", n, with_rcr=False), _energies())
    products = {"lc": lc, "lc_alt": lc_alt, "bkg": bkg}
    monkeypatch.setattr("stixcore.io.FlareListManager.STIXPYProduct", lambda path: products[path])
    fido = FakeFido(lc_paths=["lc", "lc_alt"], bkg_paths=["bkg"])

    energy = FlareListManager().add_lc_bkg_columns(
        flare_data, start=date(2024, 6, 1), end=date(2024, 7, 1), fido_client=fido
    )

    # the orphan alternate block is dropped: only the single referenced 5-row block remains
    assert len(energy) == 5
    assert {int(x) for x in energy["index"]} == {0}
    assert {int(x) for x in flare_data["energy_index"]} == {0}


# --- background candidate search (RID LUT) ------------------------------------


def _bkg_lut(rows):
    """Build a minimal RID LUT ``Table`` from ``(rid, start_iso, duration_s)`` rows."""
    t = Table()
    t["unique_id"] = [r[0] for r in rows]
    t["start_utc"] = [r[1] for r in rows]
    t["duration"] = [r[2] for r in rows]
    t["subject"] = ["BKG quiet"] * len(rows)
    t["purpose"] = ["Background"] * len(rows)
    t["comment"] = [""] * len(rows)
    return t


def test_candidates_nearest_in_time_regardless_of_side():
    lut = _bkg_lut(
        [
            (3001, "2023-06-13T00:00:00", 3600),  # past   2 d
            (3003, "2023-06-18T00:00:00", 3600),  # future 3 d
            (3002, "2023-06-05T00:00:00", 3600),  # past  10 d
            (3004, "2023-04-15T00:00:00", 3600),  # past  61 d -> outside 30 d window
        ]
    )
    t = Time("2023-06-15T00:00:00")
    cands = search_background_candidates(lut, t, window_past=30 * u.day, window_future=7 * u.day)
    # nearest start first regardless of side: 2 d past, 3 d future, 10 d past
    assert [c.rid for c in cands] == [3001, 3003, 3002]
    assert [c.side for c in cands] == ["past", "future", "past"]

    # widening the past window pulls in the legacy request, ranked by its (large) distance
    cands90 = search_background_candidates(lut, t, window_past=90 * u.day, window_future=7 * u.day)
    assert [c.rid for c in cands90] == [3001, 3003, 3002, 3004]


def test_candidates_tie_prefers_past():
    # equal distance past vs future -> past wins the tie
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600), (3003, "2023-06-20T00:00:00", 3600)])
    t = Time("2023-06-15T00:00:00")
    cands = search_background_candidates(lut, t, window_past=30 * u.day, window_future=30 * u.day)
    assert [c.rid for c in cands] == [3001, 3003]
    assert cands[0].side == "past"


def _bkg_lut_full(rows):
    """RID LUT from ``(rid, start_iso, duration_s, subject, purpose, comment)`` rows."""
    t = Table()
    t["unique_id"] = [r[0] for r in rows]
    t["start_utc"] = [r[1] for r in rows]
    t["duration"] = [r[2] for r in rows]
    t["subject"] = [r[3] for r in rows]
    t["purpose"] = [r[4] for r in rows]
    t["comment"] = [r[5] for r in rows]
    return t


def test_candidates_exclude_elevated():
    lut = _bkg_lut_full(
        [
            (1, "2023-06-14T00:00:00", 3600, "BKG elevated", "Background", ""),  # closer but excluded
            (2, "2023-06-10T00:00:00", 3600, "BKG quiet", "Background", ""),
        ]
    )
    cands = search_background_candidates(
        lut, Time("2023-06-15T00:00:00"), window_past=30 * u.day, window_future=7 * u.day
    )
    assert [c.rid for c in cands] == [2]


def test_candidates_exclude_flare_comment():
    lut = _bkg_lut_full(
        [
            (
                1,
                "2023-06-14T00:00:00",
                3600,
                "non-flaring AR?",
                "Solar Flare",
                "CL1 data request for Flare 2309081508",
            ),  # closer but flare-referenced
            (2, "2023-06-10T00:00:00", 3600, "BKG quiet", "Background", ""),
        ]
    )
    cands = search_background_candidates(
        lut, Time("2023-06-15T00:00:00"), window_past=30 * u.day, window_future=7 * u.day
    )
    assert [c.rid for c in cands] == [2]
    # keeping flare-referenced rows brings the closer one back to the front
    keep = search_background_candidates(
        lut, Time("2023-06-15T00:00:00"), window_past=30 * u.day, window_future=7 * u.day, exclude_flare_comment=False
    )
    assert [c.rid for c in keep] == [1, 2]


def test_candidates_prefer_background_purpose():
    P = 1.0 * u.day
    t = Time("2023-06-15T00:00:00")
    # subject-only match 0.5 d closer than the Background request -> Background still wins (within penalty)
    lut1 = _bkg_lut_full(
        [
            (1, "2023-06-13T00:00:00", 3600, "BKG quiet", "Background", ""),  # 2.0 d  -> eff 2.0
            (2, "2023-06-13T12:00:00", 3600, "quiet region", "obs", ""),  # 1.5 d  -> eff 2.5
        ]
    )
    c1 = search_background_candidates(lut1, t, window_past=30 * u.day, window_future=7 * u.day, purpose_penalty=P)
    assert [c.rid for c in c1] == [1, 2]
    assert c1[0].is_background

    # subject-only match clearly closer (>penalty) -> it wins
    lut2 = _bkg_lut_full(
        [
            (1, "2023-06-13T00:00:00", 3600, "BKG quiet", "Background", ""),  # 2.0 d  -> eff 2.0
            (2, "2023-06-14T12:00:00", 3600, "quiet region", "obs", ""),  # 0.5 d  -> eff 1.5
        ]
    )
    c2 = search_background_candidates(lut2, t, window_past=30 * u.day, window_future=7 * u.day, purpose_penalty=P)
    assert [c.rid for c in c2] == [2, 1]
    assert not c2[0].is_background


def test_candidates_future_window_excludes_far_future():
    lut = _bkg_lut([(4001, "2023-06-25T00:00:00", 3600)])  # 10 d in the future
    t = Time("2023-06-15T00:00:00")
    assert search_background_candidates(lut, t, window_past=30 * u.day, window_future=7 * u.day) == []
    got = search_background_candidates(lut, t, window_past=30 * u.day, window_future=14 * u.day)
    assert [c.rid for c in got] == [4001]


def test_candidates_keyword_match_only():
    lut = Table()
    lut["unique_id"] = [1, 2]
    lut["start_utc"] = ["2023-06-10T00:00:00", "2023-06-11T00:00:00"]
    lut["duration"] = [3600, 3600]
    lut["subject"] = ["Solar Flare", "some quiet interval"]  # only row 2 matches
    lut["purpose"] = ["flare", "obs"]
    lut["comment"] = ["", ""]
    cands = search_background_candidates(
        lut, Time("2023-06-15T00:00:00"), window_past=30 * u.day, window_future=7 * u.day
    )
    assert [c.rid for c in cands] == [2]


def test_candidates_empty_lut():
    assert (
        search_background_candidates(Table(), Time("2023-06-15"), window_past=30 * u.day, window_future=7 * u.day) == []
    )


def test_ridlutmanager_singleton_find_background_candidates():
    # exercises the method against the shipped test LUT (rows 3001-3004)
    cands = RidLutManager.instance.find_background_candidates(
        Time("2023-06-15T00:00:00"), window_past=30 * u.day, window_future=7 * u.day
    )
    # 3001 (5 d past) and 3003 (5 d future) are equidistant -> past wins tie, then 3002 (14 d past)
    assert [c.rid for c in cands] == [3001, 3003, 3002]


# --- find_background_file_for_time --------------------------------------------


def _cpd_name(rid):
    return f"solo_L1_stix-sci-xray-cpd_20230101T000000-20230101T010000_V01_{rid:010d}-00001.fits"


class FakeCpdFido:
    """Returns the same CPD file list for every search; the rid-in-filename filter
    inside ``find_background_file_for_time`` selects the per-candidate file."""

    def __init__(self, paths):
        self._paths = list(paths)
        self.searches = 0

    def search(self, time, instrument, data_product, level):
        self.searches += 1
        return FakeResponse(self._paths)


def _patch_cpd_products(monkeypatch, spec):
    """``spec``: {rid: rcr_array_or_None}. Builds CPD filenames + FakeProducts and
    monkeypatches STIXPYProduct to resolve them."""
    products = {}
    paths = []
    for rid, rcr in spec.items():
        name = _cpd_name(rid)
        paths.append(name)
        with_rcr = rcr is not None
        products[name] = FakeProduct(_ql_data("2023-06-10T00:00:00", 5, rcr=rcr, with_rcr=with_rcr), _energies())
    monkeypatch.setattr("stixcore.io.FlareListManager.STIXPYProduct", lambda path: products[path])
    return FakeCpdFido(paths)


def test_find_bkg_picks_closest_usable(monkeypatch):
    # 3001 starts 5 d before, 3002 starts 7 d after -> 3001 is nearest in time
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600), (3002, "2023-06-22T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.zeros(5), 3002: np.zeros(5)})
    t = Time("2023-06-15T00:00:00")
    sel = find_background_file_for_time(
        t, fido_client=fido, rid_lut=lut, min_duration=600 * u.s, require_same_elut=False
    )
    assert sel.rid == 3001  # nearest-in-time wins
    assert sel.path == Path(_cpd_name(3001))
    assert sel.valid_from == t
    # valid until the midpoint to the next later start: (2023-06-10 + 2023-06-22) / 2 = 2023-06-16
    assert sel.valid_to == Time("2023-06-16T00:00:00")


def test_find_bkg_skips_attenuator_in(monkeypatch):
    # 3001 (closer past) has attenuator in (rcr>0) -> fall through to 3002
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600), (3002, "2023-06-05T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.ones(5), 3002: np.zeros(5)})
    sel = find_background_file_for_time(
        Time("2023-06-15T00:00:00"),
        fido_client=fido,
        rid_lut=lut,
        min_duration=600 * u.s,
        require_same_elut=False,
    )
    assert sel.rid == 3002


def test_find_bkg_skips_too_short(monkeypatch):
    # 3001 requested only 300 s (< min_duration) -> skipped without even a search
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 300), (3002, "2023-06-05T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.zeros(5), 3002: np.zeros(5)})
    sel = find_background_file_for_time(
        Time("2023-06-15T00:00:00"),
        fido_client=fido,
        rid_lut=lut,
        min_duration=600 * u.s,
        require_same_elut=False,
    )
    assert sel.rid == 3002


def test_find_bkg_future_fallback(monkeypatch):
    # no past candidate -> the future one is used
    lut = _bkg_lut([(3003, "2023-06-18T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3003: np.zeros(5)})
    sel = find_background_file_for_time(
        Time("2023-06-15T00:00:00"),
        fido_client=fido,
        rid_lut=lut,
        min_duration=600 * u.s,
        require_same_elut=False,
    )
    assert sel.rid == 3003


def test_find_bkg_none_when_nothing_qualifies(monkeypatch):
    # only candidate has attenuator in -> no file, but a valid period is still returned
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.ones(5)})
    t = Time("2023-06-15T00:00:00")
    sel = find_background_file_for_time(
        t,
        fido_client=fido,
        rid_lut=lut,
        window_future=7 * u.day,
        min_duration=600 * u.s,
        require_same_elut=False,
    )
    assert sel.path is None
    assert sel.rid == -1
    assert sel.valid_from == t
    assert sel.valid_to == t + 7 * u.day  # no later candidate -> t + window_future


# --- same-ELUT criterion ------------------------------------------------------


class _FakeELUTInstance:
    def __init__(self, fn):
        self._fn = fn

    def _find_elut_file(self, dt):
        return self._fn(dt)


def _patch_elut(monkeypatch, fn):
    import types

    monkeypatch.setattr(flm_mod, "ELUTManager", types.SimpleNamespace(instance=_FakeELUTInstance(fn)))


def test_find_bkg_requires_same_elut(monkeypatch):
    # 3001 (closer past) is under a DIFFERENT ELUT than the flare; 3002 matches it
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600), (3002, "2023-06-05T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.zeros(5), 3002: np.zeros(5)})
    # flare(15)->'A', 3001(10)->'B' (different), 3002(5)->'A' (same)
    _patch_elut(monkeypatch, lambda dt: "B" if dt.day == 10 else "A")
    t = Time("2023-06-15T00:00:00")

    sel = find_background_file_for_time(
        t, fido_client=fido, rid_lut=lut, min_duration=600 * u.s, require_same_elut=True
    )
    assert sel.rid == 3002  # 3001 skipped: different ELUT configuration

    sel_off = find_background_file_for_time(
        t, fido_client=fido, rid_lut=lut, min_duration=600 * u.s, require_same_elut=False
    )
    assert sel_off.rid == 3001  # criterion off -> closest past wins


def test_find_bkg_same_elut_skipped_when_flare_elut_unknown(monkeypatch):
    # ELUT can't be resolved for the flare -> criterion cannot be enforced -> not restrictive
    lut = _bkg_lut([(3001, "2023-06-10T00:00:00", 3600)])
    fido = _patch_cpd_products(monkeypatch, {3001: np.zeros(5)})
    _patch_elut(monkeypatch, lambda dt: None)
    sel = find_background_file_for_time(
        Time("2023-06-15T00:00:00"), fido_client=fido, rid_lut=lut, min_duration=600 * u.s, require_same_elut=True
    )
    assert sel.rid == 3001


# --- add_background_file_column (valid-period cache) --------------------------


def test_add_background_file_column_uses_valid_period_cache(monkeypatch):
    calls = []

    def fake_find(time, *, fido_client, **kwargs):
        calls.append(time)
        # each selection stays valid for 2 days from the queried time
        return BackgroundSelection(path=Path("bkg.fits"), rid=42, valid_from=time, valid_to=time + 2 * u.day)

    monkeypatch.setattr(flm_mod, "find_background_file_for_time", fake_find)

    data = QTable()
    data["peak_UTC"] = Time(
        ["2023-06-15T00:00:00", "2023-06-16T00:00:00", "2023-06-18T00:00:00"]  # 3rd is beyond the 1st period
    )
    # STIXPYProduct isn't patched here; the missing "bkg.fits" is swallowed, leaving NaN spectra.
    FlareListManager().add_background_file_column(data, energy=QTable(), fido_client=object())

    assert len(calls) == 2  # 1st + 3rd flare trigger a search; 2nd reuses the cache
    assert list(data["bkg_rid"]) == [42, 42, 42]
    assert list(data["bkg_file"]) == ["bkg.fits"] * 3


# --- background-spectrum extraction ------------------------------------------


def _sci_energies_32():
    """The 32-channel science energy binning (channel/e_low/e_high in keV)."""
    lows = [
        0,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        20,
        22,
        25,
        28,
        32,
        36,
        40,
        45,
        50,
        56,
        63,
        70,
        76,
        84,
        100,
        120,
        150,
    ]
    highs = [
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        20,
        22,
        25,
        28,
        32,
        36,
        40,
        45,
        50,
        56,
        63,
        70,
        76,
        84,
        100,
        120,
        150,
        1e6,
    ]
    e = QTable()
    e["channel"] = np.arange(32, dtype=np.uint8)
    e["e_low"] = lows * u.keV
    e["e_high"] = highs * u.keV
    return e


class FakeCPD:
    """CPD stand-in: builds a data table with counts + timedel + zero triggers
    (livefrac=1) + pixel_masks (first npix pixels present) so the flux path runs."""

    def __init__(self, counts, energies, control=None, timedel_s=1.0, dsun_m=1.496e11):
        nt, ndet, npix, nE = counts.shape
        d = QTable()
        d["counts"] = counts
        d["timedel"] = np.full(nt, timedel_s) * u.s
        d["triggers"] = np.zeros((nt, 16))
        pm = np.zeros((nt, 12), dtype=np.ubyte)
        pm[:, :npix] = 1
        d["pixel_masks"] = pm
        self.data = d
        self.energies = energies
        self.control = control
        self.meta = {"DSUN_OBS": dsun_m}  # SOLO-Sun distance (m), as in a CPD primary header


class FakeCtrlProduct:
    """Minimal product exposing a control table with detector/pixel masks."""

    def __init__(self, detector_mask, pixel_mask):
        c = QTable()
        c["detector_mask"] = [np.asarray(detector_mask, dtype=np.ubyte)]
        c["pixel_mask"] = [np.asarray(pixel_mask, dtype=np.ubyte)]
        self.control = c


def test_collecting_area_cm2():
    # 30 imaging detectors, all 12 pixels -> 30 * sum(12 pixel areas)
    det = np.ones(32, dtype=np.ubyte)
    det[[8, 9]] = 0
    from stixcore.io.FlareListManager import _pixel_area_cm2

    a_full = collecting_area_cm2(FakeCtrlProduct(det, np.ones(12)))
    assert np.isclose(a_full, 30 * _pixel_area_cm2().sum())
    # 8 large pixels only -> smaller
    pm8 = np.zeros(12, dtype=np.ubyte)
    pm8[:8] = 1
    a8 = collecting_area_cm2(FakeCtrlProduct(det, pm8))
    assert np.isclose(a8, 30 * _pixel_area_cm2()[:8].sum())
    assert a8 < a_full
    # no masks -> None
    assert collecting_area_cm2(FakeCPD(np.ones((1, 32, 12, 5)) * u.ct, _energies())) is None


def test_background_spectrum_from_cpd():
    from stixcore.io.FlareListManager import _pixel_area_cm2

    nt = 5
    counts = np.ones((nt, 32, 12, 32), dtype=float)
    for t in range(nt):
        counts[t] *= t + 1  # time-varying so the median is exercised
    counts[:, 8] = 9999  # CFL detector - must be excluded
    counts[:, 9] = 9999  # BKG monitor - must be excluded
    spec, flux, e32 = background_spectrum_from_cpd(FakeCPD(counts * u.ct, _sci_energies_32(), timedel_s=1.0))
    # counts: 30 imaging dets * 12 pixels * (t+1) -> [360..1800]; median = 1080
    assert spec.shape == (32,)
    assert spec.unit == u.ct
    assert np.allclose(spec.to_value(u.ct), 1080.0)
    assert len(e32) == 32
    # flux = counts / (exp * area * dE); triggers=0 -> livefrac=1, timedel=1s
    #  exp = 30 dets * 1s = 30; area = sum(12 pixel areas)
    dE = e32["e_high"].to_value(u.keV) - e32["e_low"].to_value(u.keV)
    area = _pixel_area_cm2().sum()
    expected = 1080.0 / (30.0 * area * dE)
    assert flux.shape == (32,)
    assert flux.unit.is_equivalent(FLUX_UNIT)
    assert np.allclose(flux.to_value(FLUX_UNIT), expected, rtol=1e-6)


def test_background_spectrum_from_cpd_ragged_energies():
    # real archive CPDs telemeter a subset of channels (e.g. 20) with fewer pixels (8);
    # counts nE must stay aligned to the energies table (no forced 32-channel expansion)
    energies = _sci_energies_32()[:20]  # 20-channel binning
    counts = np.ones((4, 32, 8, 20), dtype=float)
    counts[:, 8] = 5.0  # CFL - excluded
    counts[:, 9] = 5.0  # BKG monitor - excluded
    spec, flux, e = background_spectrum_from_cpd(FakeCPD(counts * u.ct, energies))
    assert spec.shape == (20,)  # matches energies, not forced to 32
    assert len(e) == 20
    assert np.allclose(spec.to_value(u.ct), 30 * 8)  # 30 imaging dets * 8 pixels * 1 count
    assert flux.shape == (20,)
    assert np.all(np.isfinite(flux.to_value(FLUX_UNIT)))
    # rebin against a QL block works (lengths consistent) and preserves units
    c5 = rebin_spectrum_to_ql(spec, e, _energies())
    assert c5.shape == (5,)
    assert c5.unit == u.ct
    assert c5.to_value(u.ct)[0] == 6 * (30 * 8)  # 4-10 keV -> 6 sci channels
    f5 = rebin_flux_to_ql(flux, e, _energies())
    assert f5.shape == (5,)
    assert f5.unit.is_equivalent(FLUX_UNIT)
    assert np.isfinite(f5.to_value(FLUX_UNIT)[0])


def test_rebin_flux_to_ql():
    e32 = _sci_energies_32()
    # constant flux density -> dE-weighted mean is the same constant in every band
    flux = np.full(32, 3.0)
    f5 = rebin_flux_to_ql(flux, e32, _energies())
    assert np.allclose(f5, 3.0)
    # a shifted top band changes the channel set but a constant stays constant;
    # use a ramp to show band membership matters
    flux2 = np.arange(32, dtype=float)
    a = rebin_flux_to_ql(flux2, e32, _energies())
    ql2 = _energies()
    ql2["e_high"] = [10, 15, 25, 50, 100] * u.keV
    b = rebin_flux_to_ql(flux2, e32, ql2)
    assert a[4] != b[4]  # 50-84 vs 50-100 include different channels


def test_rebin_spectrum_to_ql():
    e32 = _sci_energies_32()
    counts32 = np.arange(32, dtype=float)  # channel c contributes c counts
    ql = _energies()  # QL bands read from the (energy) table, not hardcoded
    c5 = rebin_spectrum_to_ql(counts32, e32, ql)
    assert c5[0] == sum(range(1, 7))  # 4-10 keV -> ch 1..6
    assert c5[1] == sum(range(7, 12))  # 10-15   -> ch 7..11
    assert c5[2] == sum(range(12, 17))  # 15-25   -> ch 12..16
    assert c5[3] == sum(range(17, 23))  # 25-50   -> ch 17..22
    assert c5[4] == sum(range(23, 28))  # 50-84   -> ch 23..27

    # a shifted top band (50-100) must pull in channel 28 (84-100): proves it follows the table
    ql2 = _energies()
    ql2["e_high"] = [10, 15, 25, 50, 100] * u.keV
    c5b = rebin_spectrum_to_ql(counts32, e32, ql2)
    assert c5b[4] == sum(range(23, 29))
    assert c5b[4] != c5[4]


def test_add_background_file_column_extracts_and_caches(monkeypatch):
    t0 = "2024-10-01T00:00:00"
    T0 = Time(t0)
    data = QTable()
    data["flare_id"] = [1, 2, 3, 4]
    data["peak_UTC"] = Time([t0, "2024-10-01T01:00:00", "2024-10-01T02:00:00", "2024-10-01T05:00:00"])
    data["energy_index"] = np.zeros(4, dtype=np.int8)  # all reference the QL block (index 0)

    energy = _energies()
    energy["index"] = np.zeros(5, dtype=np.int8)  # QL 5-ch block, index 0

    def fake_find(time, *, fido_client, **kwargs):
        if time <= T0 + 1.5 * u.h:
            return BackgroundSelection(path="cpdA", rid=100, valid_from=T0, valid_to=T0 + 1.5 * u.h)
        if time <= T0 + 3.5 * u.h:
            return BackgroundSelection(path="cpdB", rid=200, valid_from=T0 + 1.5 * u.h, valid_to=T0 + 3.5 * u.h)
        return BackgroundSelection(path=None, rid=-1, valid_from=time, valid_to=time + 1 * u.h)

    monkeypatch.setattr(flm_mod, "find_background_file_for_time", fake_find)

    opens = []

    dsun_by_path = {"cpdA": 0.5 * 1.495978707e11, "cpdB": 0.8 * 1.495978707e11}  # m

    def fake_product(path):
        opens.append(path)
        return FakeCPD(np.ones((3, 32, 12, 32)) * u.ct, _sci_energies_32(), dsun_m=dsun_by_path[path])

    monkeypatch.setattr(flm_mod, "STIXPYProduct", fake_product)

    energy = FlareListManager().add_background_file_column(data, energy=energy, fido_client=object())

    # each CPD opened exactly once (cpdA shared by flares 0 & 1; cpdB flare 2; flare 3 has no file)
    assert opens == ["cpdA", "cpdB"]
    assert list(data["bkg_rid"]) == [100, 100, 200, -1]
    # both CPDs share the same 32-ch binning -> one new block (index 1); QL block stays index 0
    assert {int(x) for x in energy["index"]} == {0, 1}
    assert int(np.sum(np.asarray(energy["index"]) == 1)) == 32
    assert list(data["bkg_energy_index"]) == [1, 1, 1, -1]
    # counts spectra filled for in-file flares, NaN for the no-file flare
    assert data["bkg_spec"].shape == (4, 32)
    assert not np.isnan(data["bkg_spec"][0].to_value(u.ct)).any()
    assert np.isnan(data["bkg_spec"][3].to_value(u.ct)).all()
    assert np.allclose(data["bkg_spec"][0].to_value(u.ct), 360.0)  # 30 dets * 12 pix * 1
    # counts rebinned to the QL bands
    assert data["bkg_spec_ql"].shape == (4, 5)
    assert data["bkg_spec_ql"].unit == u.ct
    assert not np.isnan(data["bkg_spec_ql"][0].to_value(u.ct)).any()
    assert np.isnan(data["bkg_spec_ql"][3].to_value(u.ct)).all()
    # each QL band sums whole native channels (360 ct each), so every band is a multiple of 360
    assert np.allclose(data["bkg_spec_ql"][0].to_value(u.ct) % 360.0, 0.0)
    # flux columns (ct/s/keV/cm2), native + QL-rebinned
    assert data["bkg_spec_flux"].shape == (4, 32)
    assert data["bkg_spec_flux"].unit.is_equivalent(FLUX_UNIT)
    assert not np.isnan(data["bkg_spec_flux"][0].to_value(FLUX_UNIT)).any()
    assert np.isnan(data["bkg_spec_flux"][3].to_value(FLUX_UNIT)).all()
    assert data["bkg_spec_flux_ql"].shape == (4, 5)
    assert not np.isnan(data["bkg_spec_flux_ql"][0].to_value(FLUX_UNIT)).any()
    assert np.isnan(data["bkg_spec_flux_ql"][3].to_value(FLUX_UNIT)).all()
    # bkg_solo_sun_distance read from each CPD's DSUN_OBS header (km); NaN for the no-file flare
    d = data["bkg_solo_sun_distance"].to_value(u.AU)
    assert np.isclose(d[0], 0.5)
    assert np.isclose(d[1], 0.5)
    assert np.isclose(d[2], 0.8)
    assert np.isnan(d[3])
