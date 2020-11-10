"""

"""
from datetime import datetime
from pathlib import Path

import spiceypy as spice
from astropy.time.core import Time

SOLAR_ORBITER_NAIF_ID = -144


class SpiceManager:
    """

    """
    def __init__(self, mk_path):
        """

        Parameters
        ----------
        mk_path : str or pathlib.Path
            Path to the meta kernel

        """
        self.mk_path = Path(mk_path)
        *_, datestamp, version = self.mk_path.name.split('_')
        self.kernel_date = datetime.strptime(datestamp, '%Y%m%d')
        spice.furnsh(str(mk_path))

    @staticmethod
    def scet_to_utc(scet):
        """
        Convert SCET to UTC time strings in ISO format.

        Parameters
        ----------
        scet : `str`
            SCET time string e.g. 625237315:44104

        Returns
        -------
        `str`
            UTC time string in ISO format
        """
        # Obt to Ephemeris time (seconds past J2000)
        ephemeris_time = spice.scs2e(SOLAR_ORBITER_NAIF_ID, scet)
        # Ephemeris time to Utc
        # Format of output epoch: ISOC (ISO Calendar format, UTC)
        # Digits of precision in fractional seconds: 6
        return spice.et2utc(ephemeris_time, "ISOC", 3)

    @staticmethod
    def utc_to_scet(utc):
        """
        Convert UTC ISO format to SCET time strings.

        Parameters
        ----------
        utc : `str`
            UTC time sring in is format e.g. '2019-10-24T13:06:46.682758'

        Returns
        -------
        `str`
            SCET time string
        """
        # Utc to Ephemeris time (seconds past J2000)
        ephemeris_time = spice.utc2et(utc)
        # Ephemeris time to Obt
        return spice.sce2s(SOLAR_ORBITER_NAIF_ID, ephemeris_time)

    @staticmethod
    def scet_to_datetime(scet):
        """
        Convert SCET to datetime.

        Parameters
        ----------
        scet : `str`
            SCET time string e.g. 625237315:44104

        Returns
        -------
        `datetime.datetime`
            Datetime of SCET

        """
        et = spice.scs2e(SOLAR_ORBITER_NAIF_ID, scet)
        return spice.et2datetime(et)

    @staticmethod
    def datetime_to_scet(datetime):
        """
        Convert datetime to SCET.

        Parameters
        ----------
        datetime : datetime.datetime or astropy.time.Time
            Time to convert to SCET

        Returns
        -------
        `str`
            SCET of datetime
        """
        if isinstance(datetime, Time):
            datetime = datetime.to_datetime()
        et = spice.datetime2et(datetime)
        scet = spice.sce2s(SOLAR_ORBITER_NAIF_ID, et)
        return scet
