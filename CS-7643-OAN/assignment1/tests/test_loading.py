""" 			  		 			     			  	   		   	  			  	
Data loading Tests.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import unittest
import numpy as np
from utils import load_mnist_trainval, load_mnist_test, generate_batched_data


class TestLoading(unittest.TestCase):
    """ The class containing all test cases for this assignment"""

    def setUp(self):
        """Define the functions to be tested here."""
        pass

    def test_load_mnist(self):
        train_data, train_label, val_data, val_label = load_mnist_trainval()
        self.assertEqual(len(train_data), len(train_label))
        self.assertEqual(len(val_data), len(val_label))
        self.assertEqual(len(train_data), 4 * len(val_data))
        for img in train_data:
            self.assertIsInstance(img, list)
            self.assertEqual(len(img), 784)
        for img in val_data:
            self.assertIsInstance(img, list)
            self.assertEqual(len(img), 784)
        for t in train_label:
            self.assertIsInstance(t, int)
        for t in val_label:
            self.assertIsInstance(t, int)

    def test_generate_batch(self):
        train_data, train_label, val_data, val_label = load_mnist_trainval()
        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=128, shuffle=True, seed=1024)
        for i, b in enumerate(batched_train_data[:-1]):
            self.assertEqual(len(b), 128)
            self.assertEqual(len(batched_train_label[i]), 128)