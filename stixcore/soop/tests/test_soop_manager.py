import os
import time
import shutil

import dateutil.parser
import pytest

from stixcore.data.test import test_data
from stixcore.soop.manager import (
    FitsHeaderKeyword,
    FitsKeywordSet,
    SOOPManager,
    SoopObservationType,
)

MOVE_FILE = 'SSTX_observation_timeline_export_M04_V02.json'


@pytest.fixture
def soop_manager():
    return SOOPManager(test_data.soop.DIR)


def teardown_function():
    movedfile = test_data.soop.DIR / MOVE_FILE
    if movedfile.exists():
        os.remove(movedfile)

    movedfile = test_data.soop.DIR / (MOVE_FILE+".tmp")
    if movedfile.exists():
        os.remove(movedfile)


def test_soop_manager(soop_manager):
    assert str(soop_manager.data_root) ==\
           str(test_data.soop.DIR)


def test_soop_manager_watchdog(soop_manager):
    # 3 files are in the base dir
    assert soop_manager.filecounter == 3
    assert len(soop_manager.soops) == 8
    assert len(soop_manager.observations) == 132

    # emulate a new file approaches via rsync
    shutil.copy(test_data.soop.DIR / "wd" / MOVE_FILE, test_data.soop.DIR / (MOVE_FILE+".tmp"))
    shutil.move(test_data.soop.DIR / (MOVE_FILE+".tmp"), test_data.soop.DIR / MOVE_FILE)

    time.sleep(2)

    # the new data should be integrated now
    assert soop_manager.filecounter == 4
    assert len(soop_manager.soops) == 10
    assert len(soop_manager.observations) == 150


def test_soop_manager_find_point(soop_manager):
    start = dateutil.parser.parse("2021-10-04T12:00:00Z")
    obslist = soop_manager.find_observations(start=start)
    assert len(obslist) == 2
    for obs in obslist:
        assert obs.startDate <= start and obs.endDate >= start


def test_soop_manager_find_range(soop_manager):
    start = dateutil.parser.parse("2021-10-04T12:00:00Z")
    end = dateutil.parser.parse("2021-10-18T00:00:00Z")
    obslist = soop_manager.find_observations(start=start, end=end)
    assert len(obslist) == 16
    for obs in obslist:
        assert (obs.startDate >= start or obs.endDate >= start)
        assert (obs.endDate <= end or obs.startDate <= end)


def test_soop_manager_find_filter(soop_manager):
    start = dateutil.parser.parse("2021-10-04T12:00:00Z")
    end = dateutil.parser.parse("2021-10-18T00:00:00Z")
    obslist = soop_manager.find_observations(start=start, end=end,
                                             otype=SoopObservationType.STIX_BASIC)
    assert len(obslist) == 2


def test_soop_manager_get_keywords(soop_manager):
    start = dateutil.parser.parse("2021-10-04T12:00:00Z")
    end = dateutil.parser.parse("2021-10-18T00:00:00Z")
    keylist = soop_manager.get_fits_keywords(start=start, end=end, otype=SoopObservationType.ALL)
    assert len(keylist) == 4
    keyset = FitsKeywordSet(keylist)
    assert keyset.get(FitsHeaderKeyword(name='TARGET')).value\
        == "TBC"
    assert keyset.get(FitsHeaderKeyword(name='SOOPTYPE')).value\
        == "LF5"
    assert keyset.get(FitsHeaderKeyword(name='SOOPNAME')).value\
        == "L_FULL_LRES_MCAD_Coronal-Synoptic"
    assert keyset.get(FitsHeaderKeyword(name='OBS_ID')).value\
        .count(";") == 15


def test_soop_manager_get_keywords_time_not_found(soop_manager):
    start = dateutil.parser.parse("2016-10-04T12:00:00Z")
    end = dateutil.parser.parse("2016-10-18T00:00:00Z")
    with pytest.raises(ValueError) as e:
        _ = soop_manager.get_fits_keywords(start=start, end=end, otype=SoopObservationType.ALL)
    assert str(e.value).startswith('No soops')


def test_soop_manager_FitsKeywordSet():
    a = FitsHeaderKeyword(name="a", value="v1", comment="ca")
    a2 = FitsHeaderKeyword(name="a", value="v2", comment="ca")
    b = FitsHeaderKeyword(name="b", value="v1", comment="cb")
    b2 = FitsHeaderKeyword(name="B", value="v1", comment="cb2")

    f_set = FitsKeywordSet([a, a2, b, b2])

    f_list = f_set.to_list()
    assert len(f_list) == 2

    ma = f_set.get(a)
    assert ma.comment == "ca"
    assert ma.name == "A"
    assert ma.value == "v1;v2"

    mb = f_set.get(b)
    assert mb.comment == "cb;cb2"
    assert mb.name == "B"
    assert mb.value == "v1"
