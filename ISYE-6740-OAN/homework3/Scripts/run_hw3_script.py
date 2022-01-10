import argparse
import logging.config
from configparser import ConfigParser
from src.hw3controller import HW3Controller

logger = logging.getLogger(__name__)

def run_model(config):
    """[Run a previously trained model to retrieve a prediction based on the test data derived from
    the train/test split]

    Args:
        config ([type]): [Config parameter file for the model runs]
    """    
    try:
        #Model parameters set 
        dataset_names = config['model_run_parameters']['dataset_names'].split(',')
        num_features = config['model_run_parameters']['num_features'].split(',')
        model_names = config['model_run_parameters']['model_names'].split(',')
        controller = HW3Controller(dataset_names,num_features,model_names)
        controller.run_model()  
    except Exception as e:
        raise(e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-c", "--config", required=True, help="configuration file")
    ap.add_argument('-l', '--logging', type=str, help='Logging configuration path.')

    args = vars(ap.parse_args())

    if 'logging' in args and args['logging'] is not None:
        logging.config.fileConfig(args['logging'], disable_existing_loggers=False)

    config = ConfigParser()
    config.read(args['config'])

    try:
        run_model(config)

    except Exception as e:
        logger.error(str(e))
        raise(e)