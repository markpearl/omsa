import numpy as np
import matplotlib.pyplot as plt
import math

data = np.array([- 2.0,- 3.0,4.0,- 7.0,0.0,4.0])
theta = 0
thetas = []
thetas.append(theta)

#Start metropolis experiment and generate 10,000 random observations
for i in np.arange(1,10500+1).reshape(-1):
    theta_proposal = round(- 2 + (4 * round(np.random.rand(1)[0],4)),4)

    proposed = round(np.prod(np.exp((- 1 / 8) * (theta - theta_proposal) * (np.multiply(data,2) - theta_proposal - theta))),4)

    tau = round(((np.cos((math.pi * theta_proposal) / 4) / np.cos((math.pi * theta) / 4)) ** 2) * proposed,4)
    
    tau_min = round(np.amin(tau),4)
    if (round(np.random.rand(1)[0],4) < tau_min):
        theta = theta_proposal
    thetas.append(theta)

#Drop the first 500 observations and create the final array
thetas = np.array(thetas[500:len(thetas)])
n, bins, patches = plt.hist(x=thetas, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
print(f"Bayes estimator is the following: {np.mean(thetas)}")
#Produce the 97.5 to generate the 95% equitable set
print(f"Our 95 percent equitable set is: {np.percentile(thetas, [2.5,97.5])}")