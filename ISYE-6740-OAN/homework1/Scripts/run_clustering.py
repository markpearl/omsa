import logging
import logging.config
import argparse
from clustering.clustering_controller import ClusteringController
from imageio import imread, imsave
import numpy as np
import itertools

logger = logging.getLogger(__name__)

def run_clustering_algorithm(k: int, img: object, algorithm:str, iterations:int, img_name: str):
    """[Method for initializing k-medoids class and running algorithm]

    Arguments:
        k {int} -- [Numbers of clusters to be chosen]
        img {obj} -- [Original image that is being used for clustering]
        algorithm {str} -- [Choice of algorithm: kmeans, kmedoids or both]
        iterations {int} -- [Number of iterations to run the algorithm or until it converges]
        img_name {str} -- [Name of image]
    """
    clustering = ClusteringController(k,img,algorithm,iterations,img_name)
    logger.info("Running KMedoids algorithm for the provided image")
    points, labels, iteration = clustering.run_clustering_algorithm()
    clustering.save_image(points,labels, iteration)


if __name__ in "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()

        ap.add_argument('-l', '--logging', type=str,
                        help='Logging configuration path.')

        args = vars(ap.parse_args())

        if 'logging' in args and args['logging'] is not None:
            logging.config.fileConfig(args['logging'], disable_existing_loggers=False)

        list_configs = [
            [3,16,32], #Combinations for k
            ['./data/football.bmp','./data/beach.bmp'], #Image file paths
            ['kmedoids','kmeans'], #algorithms to run
            [1,100] #number of iterations to run
        ]
        #Running clustering algorithm against the following pictures and k-values, due to the size of the personal
        #picture only running this once
        personal_pic = [(3, './data/hockey.bmp', 'kmedoids', 1),(3, './data/hockey.bmp', 'kmeans', 1)]
        list_configs = list(itertools.product(*list_configs))
        for elem in personal_pic:
            list_configs.append(elem)

        for (k,img_name,algorithm,iterations) in list_configs:
            #Read in image and scale it's values to between 0-1
            img = imread(img_name)/255
            run_clustering_algorithm(k,img,algorithm,iterations,img_name) 

    except Exception as e:
        raise(e)