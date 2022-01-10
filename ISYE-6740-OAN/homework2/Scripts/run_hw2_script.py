import logging.config
from configparser import RawConfigParser
import argparse
from src.hw2controller import HW2Controller

logger = logging.getLogger(__name__)

def run_hw2_controller():
    """[Run method to create each associated child class for each question and get the expected result]
    """
    controller = HW2Controller()
    #Run isomap step
    controller.run_isomap()

    #Run density estimation for Q2
    controller.run_density_estimation()

    #Run gmm-model step for Q3
    controller.run_gmm_model()

if __name__ in "__main__":

    logging.basicConfig(level=logging.INFO)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('-l', '--logging', type=str,
                    help='Logging configuration path.')

    args = vars(ap.parse_args())

    if 'logging' in args and args['logging'] is not None:
        logging.config.fileConfig(args['logging'], disable_existing_loggers=False)

    try:
        run_hw2_controller()

    except Exception as e:
        raise(e)