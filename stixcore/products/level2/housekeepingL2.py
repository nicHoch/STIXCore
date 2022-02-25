"""
House Keeping data products
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

import astropy.units as u
from astropy.table import QTable

from stixcore.ephemeris.manager import Spice
from stixcore.processing.sswidl import SSWIDLProcessor, SSWIDLTask
from stixcore.products import Product
from stixcore.products.level0.housekeepingL0 import HKProduct
from stixcore.products.product import GenericPacket, L2Mixin
from stixcore.time import SCETime, SCETimeDelta, SCETimeRange

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

            data = QTable()
            data['cha_diode0'] = l2.data['hk_asp_photoa0_v']
            data['cha_diode1'] = l2.data['hk_asp_photoa1_v']
            data['chb_diode0'] = l2.data['hk_asp_photob0_v']
            data['chb_diode1'] = l2.data['hk_asp_photob1_v']
            data['time'] = [d.strftime('%Y-%m-%dT%H:%M:%S.%f')
                            for d in l2.data['time'].to_datetime()]
            data['scet_time_f'] = l2.data['time'].fine
            data['scet_time_c'] = l2.data['time'].coarse

            # TODO set to seconds
            dur = (l2.data['time'][1:] - l2.data['time'][0:-1]).as_float().value
            data['duration'] = dur[0]
            data['duration'][0:-1] = dur
            data['duration'][:] = dur[-1]

            # data['spice_disc_size'] = Spice.instance.get_sun_disc_size(date=l2.data['time'])
            data['spice_disc_size'] = [Spice.instance.get_sun_disc_size(date=d)
                                       for d in l2.data['time']]

            data['y_srf'] = 0.0
            data['z_srf'] = 0.0
            data['calib'] = 0.0
            data['error'] = ""
            data['control_index'] = l2.data['control_index']

            dataobj = dict()
            for coln in data.colnames:
                dataobj[coln] = data[coln].value.tolist()

            f = {'parentfits': str(parent),
                 'data': dataobj}

            idlprocessor[AspectIDLProcessing].params['hk_files'].append(f)

        return [l2]

    @classmethod
    def is_datasource_for(cls, *, service_type, service_subtype, ssid, **kwargs):
        return (kwargs['level'] == 'L2' and service_type == 3
                and service_subtype == 25 and ssid == 2)


class AspectIDLProcessing(SSWIDLTask):
    def __init__(self):
        script = '''

            workdir = '{{ work_dir }}'
            print, workdir
            cd, workdir

            ; I/O directories:
            ; - location of some parameter files
            param_dir = workdir + d + 'SAS_param' + d
            calib_file = param_dir + 'SAS_calib_20211005.sav'
            aperfile = param_dir + 'apcoord_FM_circ.sav'
            simu_data_file = param_dir + 'SAS_simu.sav'

            hk_files = JSON_PARSE('{{ hk_files }}', /TOSTRUCT)

            data = []
            processed_files = []
            FOREACH hk_file, hk_files, file_index DO BEGIN
                print, hk_file.parentfits
                processed_files = [processed_files, hk_file.parentfits]
                data_f = []
                for i=0L, n_elements(hk_file.DATA.cha_diode0)-1 do begin
                    data_e = { stx_aspect_dto, $
                                cha_diode0: hk_file.DATA.cha_diode0[i], $
                                cha_diode1: hk_file.DATA.cha_diode1[i], $
                                chb_diode0: hk_file.DATA.chb_diode0[i], $
                                chb_diode1: hk_file.DATA.chb_diode1[i], $
                                time: hk_file.DATA.time[i], $
                                scet_time_c: hk_file.DATA.scet_time_c[i], $
                                scet_time_f: hk_file.DATA.scet_time_f[i], $
                                duration : hk_file.DATA.duration[i], $
                                spice_disc_size : hk_file.DATA.spice_disc_size[i], $
                                y_srf : hk_file.DATA.y_srf[i], $
                                z_srf : hk_file.DATA.z_srf[i], $
                                calib : hk_file.DATA.calib[i],  $
                                error : hk_file.DATA.error[i], $
                                control_index : hk_file.DATA.control_index[i], $
                                parentfits : file_index $
                            }
                    data_f = [data_f, data_e]
                endfor

                ; START ASPECT PROCESSING

                help, data_f
                print, n_elements(data_f)

                ; save, data_f, file="/home/nicky/aspect.sav"

                print,"Calibrating data..."
                ; First, substract dark currents and applies relative gains
                calib_sas_data, data_f, calib_file

                ; Now automatically compute global calibration correction factor and applies it
                ; Note: this takes a bit of time
                auto_scale_sas_data, data_f, simu_data_file, aperfile

                print,"Computing aspect solution..."
                derive_aspect_solution, data_f, simu_data_file, interpol_r=1, interpol_xy=1

                print,"END Computing aspect solution..."
                ; END ASPECT PROCESSING

                data = [data, data_f]
            ENDFOREACH

            undefine, hk_file, hk_files, data_e, i, di, data_f, d


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
        if 'data' in result and 'processed_files' in result:
            for file_idx, resfile in enumerate(result.processed_files):
                file_path = Path(resfile.decode())
                HK = Product(file_path)

                control = HK.control
                data = QTable()

                idldata = result.data[result.data["parentfits"] == file_idx]
                n = len(idldata)

                data['time'] = SCETime(coarse=idldata['scet_time_c'], fine=idldata['scet_time_f'])
                # data['timedel'] = SCETimeDelta.from_float(idldata["duration"] * u.s)
                # we have instantaneous data so the integration time is set to 0
                data['timedel'] = SCETimeDelta(coarse=0, fine=0)
                data['time_utc'] = [t.decode() for t in idldata['time']]
                # [datetime.strptime(t.decode(), '%Y-%m-%dT%H:%M:%S.%f') for t in idldata['time']]
                data['control_index'] = idldata['control_index']
                data['spice_disc_size'] = idldata['spice_disc_size'] * u.arcsec
                data['y_srf'] = idldata['y_srf'] * u.arcsec
                data['z_srf'] = idldata['z_srf'] * u.arcsec
                # TODO do calculations
                data['solo_loc_hs'] = np.tile(np.array([1, 2, 3]), (n, 1)) * u.deg
                data['solo_loc_heeq'] = np.tile(np.array([4, 5, 6]), (n, 1)) * u.km
                data['roll_angle'] = 0 * u.deg
                control['parent'] = str(file_path.name)

                aspect = Aspect(control=control, data=data, idb_versions=HK.idb_versions)
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
