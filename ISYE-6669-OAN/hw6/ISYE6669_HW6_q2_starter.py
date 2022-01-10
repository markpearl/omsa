import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from cvxpy import *

# read in the data
data = pd.read_table('regression.dat', delim_whitespace=True, header = None)

# the number of data points and features for data splitting
nData = 100 
nFeature = 2

# data cleansing and formatting for Python 3
x = data.values[0: nData, :]
x = np.concatenate((x[0,3:].reshape((1,2)), x[1:,:nFeature]), axis=0).astype(float)
y = data.values[nData:, :]
y = np.append(y[1,3], y[2:,0])
y[-1] = 1
y = y.astype(float)