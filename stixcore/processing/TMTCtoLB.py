import logging
from time import perf_counter
from pathlib import Path

from stixcore.io.fits.processors import FitsLBProcessor
from stixcore.io.soc.manager import SOCManager
from stixcore.products.levelb.binary import LevelB
from stixcore.tmtc.packets import TMTC
from stixcore.util.logging import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

if __name__ == '__main__':
    tstart = perf_counter()
    logger.info('LevelB run')

    socm = SOCManager(Path('/Users/shane/Projects/STIX/dataview/data/tm_test'))
    out_dir = Path('/Users/shane/Projects/STIX/dataview/data/tm_test')
    out_dir.mkdir(parents=True, exist_ok=True)

    fits_processor = FitsLBProcessor(out_dir)

    files_to_process = socm.get_files(TMTC.TM)
    for tmtc_file in files_to_process:
        logger.info(f'Processing file: {tmtc_file.file}')
        # TODO sorting filter etc

        for prod in LevelB.from_tm(tmtc_file):
            if prod:
                fits_processor.write_fits(prod)

    tend = perf_counter()
    logger.info('Time taken %f', tend - tstart)
