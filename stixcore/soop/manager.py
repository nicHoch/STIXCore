import os
import re
import json
import logging
from enum import Enum
from pathlib import Path
from collections.abc import Iterable

import dateutil.parser
from intervaltree import IntervalTree
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from stixcore.util.logging import get_logger

__all__ = ['SOOPManager', 'SoopObservationType', 'FitsKeywordSet',
           'FitsHeaderKeyword', 'SoopObservation', 'SOOP']


logger = get_logger(__name__, level=logging.DEBUG)

SOOP_FILTER = "SSTX_observation_timeline_export_*.json"
SOOP_REGEX = re.compile(r'.*SSTX_observation_timeline_export_.*.json$')


class FitsHeaderKeyword:
    """Keyword object for FITS file header entries."""

    def __init__(self, *, name, value="", comment=""):
        """Creates a FitsHeaderKeyword object.

        Parameters
        ----------
        name : `str`
            the keyword name
        value : `str`, optional
            the value (gets converted to `str`), by default ""
        comment : `str`, optional
            the description (gets converted to `str`), by default ""
        """
        self.name = str(name).upper()
        self.value = str(value)
        self.comment = str(comment)

    def __eq__(self, other):
        if not isinstance(other, FitsHeaderKeyword):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __add__(self, other):
        """Combine two FitsHeaderKeyword by concanating value and comment if the name is the same.

        Parameters
        ----------
        other : `FitsHeaderKeyword`
            combine the current keyword with this one

        Returns
        -------
        `FitsHeaderKeyword`
            a new keyword object with combined values

        Raises
        ------
        ValueError
            if the other object is not of same type and addresses the same name
        """
        if self == other:
            comment = self.comment
            if self.comment != other.comment:
                comment += ";" + other.comment

            value = self.value
            if self.value != other.value:
                value += ";" + other.value

            return FitsHeaderKeyword(name=self.name, value=value, comment=comment)
        else:
            raise ValueError("Keyword must address the same name")


class FitsKeywordSet():
    """Helper object to collect and combine `FitsHeaderKeyword`."""

    def __init__(self, s=None):
        """Create a FitsKeywordSet collection.

        Parameters
        ----------
        s : `Iterable`, optional
            a list of `FitsHeaderKeyword` objects, by default None
        """
        self.dic = dict()
        if s is not None:
            self.append(s)

    def add(self, element):
        """Add a new `FitsHeaderKeyword` to the collection.

        If a same `FitsHeaderKeyword` is allready present than the new keywords
        gets combined into the old.

        Parameters
        ----------
        element : `FitsHeaderKeyword`
            the new keyword to add
        """
        if element in self.dic:
            self.dic[element] += element
        else:
            self.dic[element] = element

    def get(self, element):
        """Get the given keyword from the collection.

        Parameters
        ----------
        element : `FitsHeaderKeyword`
            a prototype keyword that should be return with the same name.

        Returns
        -------
        `FitsHeaderKeyword`
            The found keword with the same name.
        """
        return self.dic[element]

    def append(self, elements):
        """Add a a list of `FitsHeaderKeyword` to the collection.

        If a same `FitsHeaderKeyword` is allready present than the new keywords
        gets combined into the old.

        Parameters
        ----------
        element : Iterable<`FitsHeaderKeyword`>
            the new keywords to add
        """
        if isinstance(elements, Iterable):
            for e in elements:
                self.add(e)
        else:
            self.add(e)

    def to_list(self):
        """Get all keywords.

        Returns
        -------
        `list`
            list of all keywords
        """
        return [e for e in self.dic.values()]


class SoopObservationType(Enum):
    """Enum of possible observation types defines in LTP.

    This is used for Filtering.
    """
    ALL = ""
    """ Use `ALL` to address all types."""

    STIX_BASIC = "STIX_BASIC"
    """Basic operation products like HK, QL ..."""

    STIX_ANALYSIS = "STIX_ANALYSIS"
    """Bulk science data products: detector, aspect and calibration data"""


class SOOP:
    """A SOOP entry from the LTP plans."""

    def __init__(self, jsonobj):
        """Create a new SOOP object based on the generic json data.

        see: https://issues.cosmos.esa.int/solarorbiterwiki/pages/viewpage.action?pageId=44991195

        Parameters
        ----------
        jsonobj : `Object`
            the allready parsed generic object.
        """
        self.encodedSoopType = jsonobj['encodedSoopType']
        self.soopInstanceId = jsonobj['soopInstanceId']
        self.soopType = jsonobj['soopType']
        self.startDate = dateutil.parser.parse(jsonobj['startDate'])
        self.endDate = dateutil.parser.parse(jsonobj['endDate'])

    def to_fits_keywords(self):
        """Generate the fits header keywords derived from this SOOP.

        Returns
        -------
        `list`
            list of `FitsHeaderKeyword`
        """
        return list([FitsHeaderKeyword(name="TARGET", value="TBC",
                                       comment="Type of target from planning"),
                    FitsHeaderKeyword(name="SOOPTYPE", value=self.encodedSoopType,
                                      comment="Campaign ID(s) that the data belong to"),
                    FitsHeaderKeyword(name="SOOPNAME", value=self.soopType,
                                      comment="Name of the SOOP Campaign that the data belong to")])


class SoopObservation:
    """A observation entry from the LTP plans."""

    def __init__(self, jsonobj):
        """Create a new SoopObservation object based on the generic json data.

        see: https://issues.cosmos.esa.int/solarorbiterwiki/pages/viewpage.action?pageId=44991195

        Parameters
        ----------
        jsonobj : `Object`
            the allready parsed generic object.
        """
        self.comment = jsonobj['comment']
        self.compositeId = jsonobj['compositeId']
        self.socIds = jsonobj['socIds']
        self.name = jsonobj['name']
        self.type = SoopObservationType[jsonobj['name']]
        self.experiment = jsonobj['experiment']
        self.socIds = jsonobj['socIds']
        self.startDate = dateutil.parser.parse(jsonobj['startDate'])
        self.endDate = dateutil.parser.parse(jsonobj['endDate'])

    def to_fits_keywords(self):
        """Generate the fits header keywords derived from this observation.

        Returns
        -------
        `list`
            list of `FitsHeaderKeyword`
        """
        return list([FitsHeaderKeyword(name="OBS_ID", value=";".join(self.socIds),
                                       comment="Unique ID of the individual observation")])


class SOOPManager(FileSystemEventHandler):
    """Manages LTP files provided by GFTS"""

    def __init__(self, data_root):
        """Create the manager for a given data path root.

        All existing files will be index and the dir is observed.
        Newly arrived LTPs plans will be indexed on the fly.

        Parameters
        ----------
        data_root : `str` | `pathlib.Path`
            Path to the directory with all LTP files.
        """
        self.filecounter = 0
        self.data_root = data_root

    @property
    def data_root(self):
        """Get the data path root directory.

        Returns
        -------
        `pathlib.Path`
            path of the root directory
        """
        return self._data_root

    @data_root.setter
    def data_root(self, value):
        """Set the data path root.

        Parameters
        ----------
        data_root : `str` or `pathlib.Path`
            Path to the directory with all LTP files.
        """
        path = Path(value)
        if not path.exists():
            raise ValueError(f'path not found: {value}')

        self._data_root = path

        self.observer = Observer()
        self.observer.schedule(self, path,  recursive=False)
        self.observer.start()
        self.soops = IntervalTree()
        self.observations = IntervalTree()

        files = sorted(list(self._data_root.glob(SOOP_FILTER)), key=os.path.basename)

        if len(files) == 0:
            raise ValueError(f'No current SOOP files found at: {self._data_root}')

        for sfile in files:
            self._read_soop_file(sfile)

    def __del__(self):
        self.observer.stop()

    def on_moved(self, event):
        """Will be invoked if the directory abservers tracks a new arrived LTP file.

        Each new file will be scanned and indexed.
        """
        if SOOP_REGEX.match(event.dest_path):
            logger.info(f"detect new SOOP file: {event.dest_path}")
            self._read_soop_file(Path(event.dest_path))

    def find_soops(self, *, start, end=None):
        """Search for all SOOPs in the index.

        Parameters
        ----------
        start : `datetime`
            start time to look for overlapping SOOPs in utc time
        end : `datetime`, optional
            end time to look for overlapping SOOPs in utc time, by default None ()

        Returns
        -------
        `list`
            list of found `SOOP` in all indexed LTP overlapping the given timeperiod/point
        """
        intervals = set()
        if end is None:
            intervals = self.soops.at(start)
        else:
            intervals = self.soops.overlap(start, end)

        return list([o.data for o in intervals])

    def find_observations(self, *, start, end=None, otype=SoopObservationType.ALL):
        """Search for all observations in the index.

        Parameters
        ----------
        start : `datetime`
            start time to look for overlapping observations in utc time
        end : `datetime`, optional
            end time to look for overlapping observations in utc time, by default None ()
        otype : `SoopObservationType`, optional
            filter for specific type, by default SoopObservationType.ALL

        Returns
        -------
        `list`
            list of found `SOOPObservation` in all indexed LTP overlapping the given
            timeperiod/point and matching the SoopObservationType.
        """
        intervals = set()
        if end is None:
            intervals = self.observations.at(start)
        else:
            intervals = self.observations.overlap(start, end)

        if len(intervals) > 0 and otype != SoopObservationType.ALL:
            intervals = set([o for o in intervals if o.data.type == otype])

        return list([o.data for o in intervals])

    def get_fits_keywords(self, *, start, end=None, otype=SoopObservationType.ALL):
        """Searches for corresponding entries (SOOPs and Observations) in the index LTPs.

        Based on all found entries for the filter parameters a list of
        FitsHeaderKeyword is generated combining all avaliable information.

       Parameters
        ----------
        start : `datetime`
            start time to look for overlapping observations and SOOPs in utc time
        end : `datetime`, optional
            end time to look for overlapping observations and SOOPs in utc time, by default None ()
        otype : `SoopObservationType`, optional
            filter for specific type of observations, by default SoopObservationType.ALL
        Returns
        -------
        `list`
            a list of `FitsHeaderKeyword`

        Raises
        ------
        ValueError
            if no SOOPs or Observations where found in the index LTPs for the given filter settings.
        """
        kwset = FitsKeywordSet()

        soops = self.find_soops(start=start, end=end)
        if len(soops) == 0:
            raise ValueError(f"No soops found for time: {start} - {end}")
        for soop in soops:
            kwset.append(soop.to_fits_keywords())

        obss = self.find_observations(start=start, end=end, otype=otype)
        if len(obss) == 0:
            raise ValueError(f"No observations found for time: {start} - {end} : {otype}")
        for obs in obss:
            kwset.append(obs.to_fits_keywords())

        return kwset.to_list()

    def _read_soop_file(self, path):
        logger.info(f"Read SOOP file: {path}")
        with open(path) as f:
            ltp_data = json.load(f)
            for jsond in ltp_data["soops"]:
                soop = SOOP(jsond)
                self.soops.addi(soop.startDate, soop.endDate, soop)
            for jsond in ltp_data["observations"]:
                obs = SoopObservation(jsond)
                self.observations.addi(obs.startDate, obs.endDate, obs)

            self.filecounter += 1
