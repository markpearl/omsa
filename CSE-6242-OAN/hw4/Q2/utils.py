import numpy as np
from math import log, floor, ceil, log2
from collections import Counter
#export
class Utility(object):
    
    # This method computes entropy for information gain
    def entropy(self, class_y):
        # Input:            
        #   class_y         : list of class labels (0's and 1's)

        # TODO: Compute the entropy for a list of classes
        #
        # Example:
        #    entropy([0,0,0,1,1,1,1,1,1]) = 0.918 (rounded to three decimal places)

        entropy = 0
        ### Implement your code here
        #############a################################
        dict_class_vals = {}
        if len(set(class_y))>1:
            dict_class_vals[list(np.unique(class_y, return_counts=True)[0])[0]]=list(np.unique(class_y, return_counts=True)[1])[0]
            dict_class_vals[list(np.unique(class_y, return_counts=True)[0])[1]]=list(np.unique(class_y, return_counts=True)[1])[1]
            p0 = dict_class_vals[0]/len(class_y)
            p1 = dict_class_vals[1]/len(class_y)
            entropy = -p0*log2(p0)-p1*log2(p1)
        else:
            entropy=0
        #############################################
        return entropy


    def partition_classes(self, X, y, split_attribute, split_val):
        # Inputs:
        #   X               : data containing all attributes
        #   y               : labels
        #   split_attribute : column index of the attribute to split on
        #   split_val       : a numerical value to divide the split_attribute

 

        # TODO: Partition the data(X) and labels(y) based on the split value - BINARY SPLIT.
        # 
        # Split_val should be a numerical value
        # For example, your split_val could be the mean of the values of split_attribute
        #
        # You can perform the partition in the following way
        # Numeric Split Attribute:
        #   Split the data X into two lists(X_left and X_right) where the first list has all
        #   the rows where the split attribute is less than or equal to the split value, and the 
        #   second list has all the rows where the split attribute is greater than the split 
        #   value. Also create two lists(y_left and y_right) with the corresponding y labels.



        '''
        Example:

 

        X = [[3, 10],                 y = [1,
             [1, 22],                      1,
             [2, 28],                      0,
             [5, 32],                      0,
             [4, 32]]                      1]

 

        Here, columns 0 and 1 represent numeric attributes.

 

        Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
        Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

 

        X_left = [[3, 10],                 y_left = [1,
                  [1, 22],                           1,
                  [2, 28]]                           0]

 

        X_right = [[5, 32],                y_right = [0,
                   [4, 32]]                           1]

 

        ''' 

        X_left = []
        X_right = []

        y_left = []
        y_right = []
        ### Implement your code here
        #############################################
        for x_elem,y_elem in zip(X,y):
            if x_elem[split_attribute]<= split_val:
                X_left.append(x_elem)
                y_left.append(y_elem)
            else:
                X_right.append(x_elem)
                y_right.append(y_elem)
        
        #############################################
        return (X_left, X_right, y_left, y_right)


    def information_gain(self, previous_y, current_y):
        # Inputs:
        #   previous_y: the distribution of original labels (0's and 1's)
        #   current_y:  the distribution of labels after splitting based on a particular
        #               split attribute and split value

        # TODO: Compute and return the information gain from partitioning the previous_y labels
        # into the current_y labels.
        # You will need to use the entropy function above to compute information gain
        # Reference: http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf

        """
        Example:

        previous_y = [0,0,0,1,1,1]
        current_y = [[0,0], [1,1,1,0]]

        info_gain = 0.45915
        """

        info_gain = 0
        ### Implement your code here
        #############################################
        H = self.entropy(previous_y)
        HL = self.entropy(current_y[0])
        PL = round(len(current_y[0])/len(previous_y),3)
        HR = self.entropy(current_y[1])
        PR = round(len(current_y[1])/len(previous_y),3)
        
        info_gain = round(H - (HL*PL+HR*PR),4)
        #############################################
        return info_gain


    def best_split(self, X, y):
        # Inputs:
        #   X       : Data containing all attributes
        #   y       : labels
        #   TODO    : For each node find the best split criteria and return the split attribute, 
        #             spliting value along with  X_left, X_right, y_left, y_right (using partition_classes) 
        #             in the dictionary format {'split_attribute':split_attribute, 'split_val':split_val, 
        #             'X_left':X_left, 'X_right':X_right, 'y_left':y_left, 'y_right':y_right, 'info_gain':info_gain}
        '''

        Example: 

        X = [[3, 10],                 y = [1, 
             [1, 22],                      1, 
             [2, 28],                      0, 
             [5, 32],                      0, 
             [4, 32]]                      1] 

        Starting entropy: 0.971 

        Calculate information gain at splits: (In this example, we are testing all values in an 
        attribute as a potential split value, but you can experiment with different values in your implementation) 

        feature 0:  -->    split_val = 1  -->  info_gain = 0.17 
                           split_val = 2  -->  info_gain = 0.01997 
                           split_val = 3  -->  info_gain = 0.01997 
                           split_val = 4  -->  info_gain = 0.32 
                           split_val = 5  -->  info_gain = 0 
                           
                           best info_gain = 0.32, best split_val = 4 


        feature 1:  -->    split_val = 10  -->  info_gain = 0.17 
                           split_val = 22  -->  info_gain = 0.41997 
                           split_val = 28  -->  info_gain = 0.01997 
                           split_val = 32  -->  info_gain = 0 

                           best info_gain = 0.4199, best split_val = 22 

 
       best_split_feature: 1  
       best_split_val: 22  

       'X_left': [[3, 10], [1, 22]]  
       'X_right': [[2, 28],[5, 32], [4, 32]]  

       'y_left': [1, 1]  
       'y_right': [0, 0, 1] 
        '''
        
        split_attribute = 0
        split_val = 0
        feature_vals = {}
        X_left, X_right, y_left, y_right = [], [], [], []
        ### Implement your code here
        #############################################
        num_features = len(X[0])
        for feature_num in range(num_features-1):
            information_gain = []
            nested_dict = {}
            distinct_vals = list(set(sorted(list(list(zip(*X))[feature_num]))))
            for idx,split_value in enumerate(distinct_vals):
                X_l, X_r, y_l, y_r = self.partition_classes(X,y,feature_num,split_value)
                ig = self.information_gain(y,[y_l,y_r])
                information_gain.append(ig)
            index_max_ig = information_gain.index(max(information_gain))
            nested_dict['split_val']=distinct_vals[index_max_ig]
            nested_dict['information_gain']=max(information_gain)
            feature_vals[f'feature_{feature_num}']=nested_dict
        best_feature=max(feature_vals, key=lambda v: feature_vals[v]['information_gain'])
        best_feature_val = int(best_feature.split('_')[1])
        best_split_val = feature_vals[best_feature]['split_val']
        X_left, X_right, y_left, y_right = self.partition_classes(X,y,best_feature_val,best_split_val)
        info_gain = feature_vals[best_feature]['information_gain']
        final_dict = {
            'split_attribute':best_feature_val, 
            'split_val':best_split_val, 
            'X_left':X_left, 
            'X_right':X_right, 
            'y_left':y_left, 
            'y_right':y_right, 
            'info_gain':info_gain}
        return final_dict
        #############################################