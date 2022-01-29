from datetime import datetime
#export
import csv
import numpy as np  # http://www.numpy.org
import ast
from math import log, floor, ceil
import random
import numpy as np
from utils import Utility
from collections import Counter

#export
class DecisionTree(object):
    def __init__(self, max_depth):
        # Initializing the tree as an empty dictionary or list, as preferred
        self.tree = {}
        self.max_depth = max_depth
        
    	
    def learn(self, X, y, par_node = {}, depth=0):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in Utility class to train the tree
        
        # par_node is a parameter that is useful to pass additional information to call 
        # the learn method recursively. Its not mandatory to use this parameter

        # Use the function best_split in Utility class to get the best split and 
        # data corresponding to left and right child nodes
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        ### Implement your code here
        #############################################
        utils = Utility()
        num_samples_per_class = [np.sum(np.array(y) == i) for i in range(len(set(y)))]
        value = np.argmax(num_samples_per_class)
        node = Node(
            value = value
        )
        
        if depth < self.max_depth:
            result_dict =utils.best_split(X,y)
            node.feature=result_dict['split_attribute']
            node.threshold=result_dict['split_val']
            node.entropy = utils.entropy(y)
            self.tree[f'node_{len(self.tree)+1}']=node
            #Do recursive call for both lead nodes
            if node.entropy != 0:
                new_depth = depth+1
                node.left = self.learn(result_dict['X_left'],result_dict['y_left'],None,new_depth)
                node.right = self.learn(result_dict['X_right'],result_dict['y_right'],None,new_depth)
            
        return node
        #############################################


    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        ### Implement your code here
        #############################################
        node = self.tree['node_1']
        while node.left:
            if record[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None,entropy=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.entropy=entropy
    

        #############################################
    # def _traverse_tree(self, x, node):
    #     if node.is_leaf_node():
    #         return node.value

    #     if x[node.feature] <= node.threshold:
    #         return self._traverse_tree(x, node.left)
    #     return self._traverse_tree(x, node.right)