import argparse
import logging.config
from configparser import ConfigParser
from src.controller import Controller

logger = logging.getLogger(__name__)

def run_pipeline(config):
    """[Run a previously trained model to retrieve a prediction based on the test data derived from
    the train/test split]

    Args:
        config ([type]): [Config parameter file for the model runs]
    """    
    try:
        #Model parameters set 
        dataset_names = config['model_run_parameters']['dataset_names'].split(',')
        ##num_features = config['model_run_parameters']['num_features'].split(',')
        model_names = config['model_run_parameters']['model_names'].split(',')
        time_col_names = config['model_run_parameters']['time_col_names'].split(',')
        repeats = int(config['model_run_parameters']['repeats'])
        splits = int(config['model_run_parameters']['splits'])
        controller = Controller(dataset_names,model_names,time_col_names)
        X_train = controller.run_feature_processing(ordinal_encoding=True, one_hot_ecoding=True, corr_features=False)
        controller.run_model(X_train,controller.Y)

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
        run_pipeline(config)

    except Exception as e:
        logger.error(str(e))
        raise(e)