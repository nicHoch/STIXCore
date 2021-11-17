"""
House Keeping data products
"""
import json
import tempfile
from pathlib import Path
from collections import defaultdict

from astropy.table import QTable

from stixcore.processing.sswidl import SSWIDLProcessor, SSWIDLTask
from stixcore.products import Product
from stixcore.products.level0.housekeepingL0 import HKProduct
from stixcore.products.product import GenericPacket, L2Mixin
from stixcore.time import SCETimeRange

__all__ = ['MiniReport', 'MaxiReport', 'Aspect']


class MiniReport(HKProduct, L2Mixin):
    """Mini house keeping reported during start up of the flight software.

    In level 2 format.
    """

    def __init__(self, *, service_type, service_subtype, ssid, control, data,
                 idb_versions=defaultdict(SCETimeRange), **kwargs):
        super().__init__(service_type=service_type, service_subtype=service_subtype,
                         ssid=ssid, control=control, data=data,
                         idb_versions=idb_versions, **kwargs)
        self.name = 'mini'
        self.level = 'L2'
        self.type = 'hk'

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return (kwargs['level'] == 'L2' and service_type == 3
                and service_subtype == 25 and ssid == 1)


class MaxiReport(HKProduct, L2Mixin):
    """Maxi house keeping reported in all modes while the flight software is running.

        In level 2 format.
    """

    def __init__(self, *, service_type, service_subtype, ssid, control, data,
                 idb_versions=defaultdict(SCETimeRange), **kwargs):
        super().__init__(service_type=service_type, service_subtype=service_subtype,
                         ssid=ssid, control=control, data=data, idb_versions=idb_versions, **kwargs)
        self.name = 'maxi'
        self.level = 'L2'
        self.type = 'hk'

    @classmethod
    def from_level1(cls, l1product, idbm=GenericPacket.idb_manager, parent='', idlprocessor=None):
        l2 = cls(service_type=l1product.service_type,
                 service_subtype=l1product.service_subtype,
                 ssid=l1product.ssid,
                 control=l1product.control,
                 data=l1product.data,
                 idb_versions=l1product.idb_versions)

        l2.control.replace_column('parent', [parent.name] * len(l2.control))
        l2.fits_header = l1product.fits_header

        if isinstance(idlprocessor, SSWIDLProcessor):

            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile = Path(tfile.name)

            f = {'inpath': str(parent.parent),
                 'inname': parent.name,
                 'outpath': str(tfile.parent),
                 'outname': tfile.name + ".fits",
                 'error': None}

            idlprocessor[AspectIDLProcessing].params['hk_files'].append(f)

        return [l2]

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return (kwargs['level'] == 'L2' and service_type == 3
                and service_subtype == 25 and ssid == 2)


class AspectIDLProcessing(SSWIDLTask):
    def __init__(self):
        script = '''

            common config, param_dir, def_calibfile, data_dir, out_dir, sas_version

            workdir = '{{ work_dir }}'
            print, workdir
            cd, workdir

            sas_version = '2021-08-09'

            ; I/O directories:
            ; - location of some parameter files
            param_dir = workdir + d + 'SAS_param' + d

            ; - default file for calibration (contains relative gains and biases)
            def_calibfile = 'SAS_calib_2020'


            ; Path to the SPICE kernels
            spice_dir = "{{ spice_dir }}"
            spice_kernel = "{{ spice_kernel }}"

            ; Load the Spice kernels that we need
            cspice_kclear
            add_sunspice_mission, 'solo'
            load_sunspice_solo

            cd, spice_dir
            cspice_furnsh, spice_dir + spice_kernel

            hk_files = JSON_PARSE('{{ hk_files }}', /TOARRAY, /TOSTRUCT)
            generatedfiles = []

            FOREACH hk_file, hk_files DO BEGIN

                ; - location of the L1 HK data files
                data_dir = hk_file.INPATH + d
                ; - Output directory
                out_dir = hk_file.OUTPATH + d

                in_file = hk_file.INNAME

                print,"Reading L1 data file: " + in_file

                data = read_hk_data(in_file, quiet=0)

                print,"Calibrating data..."
                cal_factor = 1.10   ; roughly OK for 2021 data
                calib_sas_data, data, factor=cal_factor

                print,"processing data..."
                process_SAS_data, in_file, cal_factor=cal_factor, outfile=hk_file.OUTNAME

                generatedfiles = [generatedfiles, hk_file]
            ENDFOREACH

            undefine, data, hk_file, hk_files

'''
        spice_dir = '/data/stix/spice/kernels/mk/'
        spice_kernel = 'solo_ANC_soc-flown-mk_V107_20211028_001.tm'

        super().__init__(script=script, work_dir='stix/idl/processing/aspect/',
                         params={'hk_files': list(),
                                 'spice_dir': spice_dir,
                                 'spice_kernel': spice_kernel})

    def pack_params(self):
        packed = self.params.copy()
        packed['hk_files'] = json.dumps(packed['hk_files'])
        return packed

    def postprocessing(self, result, fits_processor):
        files = []

        if 'generatedfiles' in result:
            for resfile in result.generatedfiles:
                file_path = Path(resfile.outpath.decode()) / resfile.outname.decode()
                control_as = QTable.read(file_path, hdu='CONTROL')
                data_as = QTable.read(file_path, hdu='DATA')

                file_path_in = Path(resfile.inpath.decode()) / resfile.inname.decode()
                HK = Product(file_path_in)

                as_range = range(0, len(data_as))

                data_as['time'] = HK.data['time'][as_range]
                data_as['timedel'] = HK.data['timedel'][as_range]
                data_as['control_index'] = HK.data['control_index'][as_range]
                control_as['index'] = HK.control['index'][as_range]
                control_as['raw_file'] = HK.control['raw_file'][as_range]
                control_as['parent'] = resfile.inname.decode()

                del data_as['TIME']

                aspect = Aspect(control=control_as, data=data_as, idb_versions=HK.idb_versions)
                files.extend(fits_processor.write_fits(aspect))

        return files


class Aspect(HKProduct, L2Mixin):
    """Aspect auxiliary data.

    In level 2 format.
    """

    def __init__(self, *, service_type=0, service_subtype=0, ssid=1, control, data,
                 idb_versions=defaultdict(SCETimeRange), **kwargs):
        super().__init__(service_type=service_type, service_subtype=service_subtype,
                         ssid=ssid, control=control, data=data,
                         idb_versions=idb_versions, **kwargs)
        self.name = 'aspect'
        self.level = 'L2'
        self.type = 'aux'

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return (kwargs['level'] == 'L2' and service_type == 0
                and service_subtype == 0 and ssid == 1)