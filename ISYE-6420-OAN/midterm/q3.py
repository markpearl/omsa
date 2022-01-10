import numpy as np
from scipy.special import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.RandomState(1)

#Set the number of iterations
n = 197
#Set the empty lists to hold the updated values for theta and tau for each variable (i.e 1,2)
thetas1 = []
y = []
a = 35
b = 39

# start, initial values
y0 = 0
theta1 = 0

for i in np.arange(1,n+1).reshape(-1):
    #Determine the value of updated value
    updatedTheta1 = np.random.beta(theta1+a,b, size=1)[0]
    p  = updatedTheta1 / (2+updatedTheta1)
    updatedY0 = np.random.binomial(n, p, 1)[0]
    thetas1.append(updatedTheta1)   
    theta1 = updatedTheta1

print(np.mean(thetas1))
n, bins, patches = plt.hist(x=thetas1,density=True, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
#Produce the 97.5 to generate the 95% equitable set
print(f"Our 95 percent equitable set is: {np.percentile(thetas1, [2.5,97.5])}")