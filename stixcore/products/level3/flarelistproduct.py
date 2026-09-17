import numpy as np
from sunpy.time.timerange import TimeRange

from stixcore.ephemeris.manager import Spice
from stixcore.products.product import GenericProduct, L3Mixin
from stixcore.time.datetime import SCETime, SCETimeRange
from stixcore.util.logging import get_logger

__all__ = ["FlareListProduct", "PeakPreviewImage"]

logger = get_logger(__name__)


class FlareListProduct(GenericProduct, L3Mixin):
    """Product not based on direct TM data but on time ranges defined in flare lists.

    In level 3 format.
    """

    @classmethod
    def from_timerange(cls, timerange: SCETimeRange, *, flarelistparent: str = ""):
        """Create a product for the given time range and parent flare list.

        Parameters
        ----------
        timerange : `~stixcore.time.datetime.SCETimeRange`
            The time range the product covers.
        flarelistparent : str, optional
            Identifier of the parent flare list.

        Notes
        -----
        Not yet implemented (stub).
        """
        pass


class PeakPreviewImage(FlareListProduct):
    """Per-flare peak-preview image product (``Name="peakpreviewimg"``, ssid 5, level L3).

    Holds the CLEAN back-projection `~sunpy.map.Map` images reconstructed around a flare peak
    (see `~stixcore.products.level3.flarelist.FlarePeakPreviewMixin`), together with the parent
    flare-list product(s) they were derived from. One instance is written per flare.
    """

    PRODUCT_PROCESSING_VERSION = 1
    Level = "L3"
    Type = "sci"
    Name = "peakpreviewimg"

    def __init__(self, control, data, energy, maps, parents, *, product_name_suffix="", **kwargs):
        """Build the product from its tables, reconstructed maps and parent product(s)."""
        super().__init__(service_type=0, service_subtype=0, ssid=5, control=control, data=data, energy=energy, **kwargs)
        self.name = f"{PeakPreviewImage.Name}-{product_name_suffix}"
        self.level = PeakPreviewImage.Level
        self.type = PeakPreviewImage.Type
        self.energy = energy
        self.maps = maps
        self.parents = parents

        self.add_additional_header_keyword(("NR_MAPS", len(maps) if maps else 0, "number of maps in file"))

    @property
    def parent(self):
        """The parent flare-list product(s) this image was derived from, as a 1-d array."""
        return np.atleast_1d(self.parents)

    @property
    def utc_timerange(self):
        """The preview time range in UTC (`~sunpy.time.TimeRange`)."""
        return TimeRange(self.data["preview_start_UTC"][0], self.data["preview_end_UTC"][0])

    @property
    def scet_timerange(self):
        """The preview time range in spacecraft elapsed time (approximated via Spice)."""
        tr = self.utc_timerange
        logger.warning(
            "scet_timerange will be approximated using Spice. Better to work with utc_timerange property to avoid automatic time conversion"
        )
        start = SCETime.from_string(Spice.instance.datetime_to_scet(tr.start)[2:])
        end = SCETime.from_string(Spice.instance.datetime_to_scet(tr.end)[2:])
        return SCETimeRange(start=start, end=end)

    def split_to_files(self):
        """Yield the product(s) to write; a peak-preview image is always a single file."""
        return [self]

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        """Whether this class handles the given L3 / ssid-5 product identifiers."""
        return kwargs["level"] == PeakPreviewImage.Level and service_type == 0 and service_subtype == 0 and ssid == 5
