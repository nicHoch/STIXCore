.. _flarelist_sdc:

****************************
SDC Flare List (Level 3)
****************************

The SDC flare list is an enriched, level-3 flare catalogue. It starts from the operational
flare list produced by the STIX Data Center (SDC) and adds STIX-derived quantities that the
source list does not provide reliably: quicklook lightcurve / background counts at the flare
peak, a quiet-time background spectrum, and — for the higher product levels — a flare
location and peak-preview images.

This page describes where the raw data comes from and each enrichment step, including the
decisions taken when several inputs are available (which background period, which CPD file,
which imaging parameters). For the class and column reference see
:doc:`/code_ref/products/l3_flarelist` (`~stixcore.products.level3.flarelist`,
`~stixcore.products.level3.flarelistproduct`) and :doc:`/code_ref/io`
(`~stixcore.io.FlareListManager`).

.. note::

   A parallel ``FlarelistSC*`` family (a flare list detected inside STIXCore rather than
   mirrored from the SDC) is planned but not yet in use; it is intentionally omitted here.


Product chain
=============

The SDC products form a chain in which each level adds one enrichment step (built by
`~stixcore.processing.FlareListL3.FlareListL3` for the first level and upgraded
level-to-level by `~stixcore.processing.FLtoFL.FLtoFL`):

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Product
     - ssid
     - Adds
   * - `~stixcore.products.level3.flarelist.FlarelistSDC`
     - 2
     - Base list: id, times, GOES class/flux, QL peak & quiet-time background, SOOP campaign.
   * - `~stixcore.products.level3.flarelist.FlarelistSDCLocation`
     - 3
     - Flare location from CPD imaging (+ SOLO position, imaging-quality metric, 1-AU fluxes).
   * - `~stixcore.products.level3.flarelist.FlarelistSDCLocationImage`
     - 4
     - Per-flare peak-preview CLEAN images (`~stixcore.products.level3.flarelistproduct.PeakPreviewImage`).


Raw data in — the STIX Data Center flare list
=============================================

`~stixcore.io.FlareListManager.SDCFlareListManager` maintains a local CSV mirror of the SDC
flare list. ``read_flarelist`` fetches it with ``stixdcpy.fetch_flare_list(start, end)`` in
roughly monthly chunks from 2020-01-01 up to *now* — the API is batched by month and
throttled (~2 calls/s, so the loop sleeps between chunks), and incremental updates re-fetch
the last ~60 days to catch late revisions. The combined list is de-duplicated, sorted by
``peak_UTC`` and cached to CSV. ``get_data`` then slices out the requested month.

The following source fields are consumed (see
`~stixcore.io.FlareListManager.SDCFlareListManager.get_data`):

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Source (CSV) field
     - Product column
   * - ``flare_id``
     - ``flare_id``
   * - ``start_UTC``, ``duration``, ``end_UTC``, ``peak_UTC``
     - same names (UTC time columns / duration in s)
   * - ``GOES_class``, ``goes_estimated_{min,max,mean}_class``
     - ``GOES_class``, ``goes_{min,max,mean}_class_est``
   * - ``GOES_flux``, ``goes_estimated_{min,max,mean}_flux``
     - ``GOES_flux``, ``goes_{min,max,mean}_flux_est`` (stored as ``10**value`` W/m²)

.. warning::

   The ``GOES_*`` columns describe the Earth-viewed GOES/XRS flux and are **not** derived
   from STIX — they are meaningless when the flare is occulted from Earth. The source list's
   own at-peak lightcurve counts and attenuator flag are **unreliable** and are deliberately
   re-derived from real STIX data (next section).


Enrichment step — QL lightcurve & background at peak
====================================================

`~stixcore.io.FlareListManager.FlareListManager.add_lc_bkg_columns` builds one monthly
timeline each for the ``ql_lightcurve`` and ``ql_background`` products (via
`~stixcore.io.FlareListManager.build_month_timeline`, so the daily files are opened only
once), then for every flare picks the bin nearest the peak
(`~stixcore.io.FlareListManager.nearest_bin_index`). It records, for the five QL energy
channels, the raw counts (``lc_peak``, ``lc_bkg_peak``) and the livetime- and
area-normalized flux (``lc_peak_flux``, ``lc_bkg_peak_flux``, in
``ct s⁻¹ keV⁻¹ cm⁻²``), together with ``rcr_at_peak`` / ``rcr_max``, the derived ``att_in``
flag and the ``energy_index`` into the energy table. This replaces the unreliable
source-CSV values with quantities taken straight from the L1 quicklook data.


Enrichment step — quiet-time background file
============================================

For the background *spectrum* a quiet-time science file has to be chosen.
`~stixcore.io.FlareListManager.find_background_file_for_time` implements the selection:

* **Candidate ranking.** Background requests are taken from the RID look-up table
  (``search_background_candidates``) and ranked **nearest-in-time first** — the request whose
  start is closest to the flare, past or future — within separate look-back / look-ahead
  windows (``window_past``, default 30 d; ``window_future``, default 7 d).
* **Acceptance checks.** A candidate is accepted only if its requested integration is long
  enough (``min_duration``, default 1200 s), its real ``sci_xray_cpd`` L1 file can be resolved
  (and the filename's request id matches), and the **attenuator is out for the whole file**
  (``rcr == 0``). The first candidate that passes wins.
* **Optional stricter filters** (all default *off* except same-ELUT, because they move common
  cases rather than edge cases), each backed by a ``[Processing]`` config key:

  .. list-table::
     :header-rows: 1
     :widths: 55 45

     * - Filter
       - Config key
     * - Require the same ELUT as the flare time
       - ``flarelist_bkg_require_same_elut`` (default on)
     * - Prefer ``purpose == "Background"`` requests
       - ``flarelist_bkg_purpose_penalty_days``
     * - Drop "elevated" backgrounds
       - ``flarelist_bkg_exclude_elevated``
     * - Drop requests whose comment references a specific flare
       - ``flarelist_bkg_exclude_flare_comment``

* **Validity-interval caching.** The result carries a ``[valid_from, valid_to]`` interval, so
  the monthly (time-ordered) loop reuses one selection for every later flare until a different
  candidate would become effectively closer (or the window edge is reached). This is why the
  same background file is not re-resolved flare by flare.

The spectrum itself (`~stixcore.io.FlareListManager.background_spectrum_from_cpd`) is the
**median over the file's time bins** — the "most recent quiet period" — summed over the 30
imaging detectors and their pixels. It is stored as native-channel counts (``bkg_spec``) and
flux (``bkg_spec_flux``), and rebinned to the QL lightcurve bands as counts (``bkg_spec_ql``)
and flux (``bkg_spec_flux_ql``); the file's native (32-channel) binning is appended to the
energy table and referenced by ``bkg_energy_index``.


Enrichment step — flare location
================================

`~stixcore.products.level3.flarelist.FlarePositionMixin.add_flare_position` (products at
ssid ≥ 3) adds a location for each flare. Per flare it searches for the daily ancillary
``asp_ephemeris`` file (cached per day) and for ``sci_xray_cpd`` files over
``[start, end]``, recording ``anc_ephemeris_path`` and ``cpd_path``.

CPD-file selection
------------------

When several CPD files cover a flare, each candidate is read as a full product
(`~stixpy.product.Product`) and scored on four criteria (in priority order)::

    cpd_res.sort(["inc_peak", "inc_flare", "_neg_min_dt", "ebins"], reverse=True)   # take the top row

* ``inc_peak`` — whether the flare **peak** time falls inside the file (preferred first),
* ``inc_flare`` — **percentage** of the flare ``start..end`` duration covered by the file (higher preferred next),
* **min time resolution** — the shortest ``timedel`` among the data-table bins overlapping the flare
  (**shorter is better**; sorted via its negative ``_neg_min_dt``, in deciseconds),
* ``ebins`` — number of **energy bins** in the energy table (**more is better**).

(A ``TODO`` notes more criteria may be added.) **The CPD file chosen here is the same file later used to
make the peak-preview images**, so the choice serves both the location and the imaging.

Fit time & energy range
-----------------------

The fit uses ``[peak − 20 s, peak + 20 s]`` clamped to ``[start, end]`` (falling back to the
file's own time range if there is no overlap). If ``rcr`` is not constant across that window,
the window is widened to ±40 s and the **longest constant-``rcr`` sub-sequence**
(`~stixcore.products.level3.flarelist.longest_constant_sequence`) is used instead. The energy
range is ``[4, 16] keV``, widening to ``[4, 25] keV`` when the attenuator is in
(``rcr > 0``).

Imaging parameters
------------------

The location is estimated by
`~stixcore.products.level3.processing.stx_estimate_flare_location` (a port of the STIX-GSW
IDL routine of the same name):

* meta pixels via ``create_meta_pixels(no_shadowing=True, flare_location=[0, 0]″)``, then
  ``create_visibility`` and ``calibrate_visibility`` with the **Sun centre** as phase centre;
* only the **coarse sub-collimators 7–10** are used
  (``isc = [3, 20, 22, 16, 14, 32, 21, 26, 4, 24, 8, 28]``);
* a back-projection map of ``512 × 512`` pixels with plate scale
  ``pixel = rsun_obs · 2.6 / imsize`` (the factor 2.6 keeps the full solar disc in the field
  of view, matching the IDL implementation); the flare location is the brightest pixel,
  transformed to Helioprojective coordinates (a spherical screen is assumed for off-disk
  positions);
* an imaging-quality metric ``sidelobes_ratio`` is computed as the strongest sidelobe outside
  a 200″ radius relative to the peak — a value close to (≳ 0.9) or above the peak indicates the
  location is unreliable. It is surfaced as the ``sidelobes_ratio`` column.

.. note::

   The peak-preview *image* product
   (`~stixcore.products.level3.flarelist.FlarePeakPreviewMixin`, ssid 4) runs a fuller CLEAN
   reconstruction in the ``[4, 20]`` and ``[20, 120] keV`` bands on the **same selected CPD
   file**. Its output columns are not yet finalised and are out of scope for this page.


Outputs
=======

The per-flare columns (units, shapes, meanings) are documented as the column descriptions in
the API reference — see `~stixcore.products.level3.flarelist` and
`~stixcore.io.FlareListManager`. A consolidated data dictionary is planned as a follow-up.
