import numpy as np
from scipy.special import gamma
from scipy.stats import norm

np.random.RandomState(1)
y1 = np.array([134.0,146.0,104.0,119.0,124.0,161.0,107.0,83.0,113.0,129.0,97.0,123.0])
n1 = len(y1)
y2 = np.array([70.0,118.0,101.0,85.0,107.0,132.0,94.0])
n2 = len(y2)    

#Set the number of iterations
n = 197
#Set the empty lists to hold the updated values for theta and tau for each variable (i.e 1,2)
thetas1 = []
taus1 = []
thetas2 = []
taus2 = []

#Generate the sum variables used for the below loop
sumy1 = sum(y1)
sumy2 = sum(y2)

#Set the initial values required 
theta10 = 110
tau10 = 1 / 100
theta20 = 110
tau20 = 1 / 100
a1 = 0.01
b1 = 4
a2 = 0.01
b2 = 4
# start, initial values
theta1 = 110
tau1 = 1 / 100

theta2 = 110
tau2 = 1 / 100

for i in np.arange(1,n+1).reshape(-1):
    #Determine the value of updated value
    updatedTheta1 = np.sqrt(1 / (tau10 + 1 * tau1)) * norm.ppf(np.random.rand(1))[0] + (tau1 * sumy1 + tau10 * theta10) / (tau10 + n1 * tau1)
    p1 = (b1 + 1 / 2) * sum((y1 - updatedTheta1) ** 2)
    updatedTau1 = np.random.gamma(a1 + n1 / 2,1 / p1)
    thetas1.append(updatedTheta1)   
    taus1.append(updatedTau1)
    theta1 = updatedTheta1
    tau1 = updatedTau1
    updatedTheta2 = np.sqrt(1 / (tau20 + 1 * tau2)) * norm.ppf(np.random.rand(1))[0] + (tau2 * sumy2 + tau20 * theta20) / (tau20 + n2 * tau2)
    p2 = b2 + 1 / 2 * sum((y2 - updatedTheta2) ** 2)
    updatedTau2 = np.random.gamma(a2 + n2 / 2,1 / p2)
    thetas2.append(updatedTheta2)
    taus2.append(updatedTau2)
    theta2 = updatedTheta2
    tau2 = updatedTau2


thetas1 = np.array(thetas1[500:len(thetas1)+1])
thetas2 = np.array(thetas2[500:len(thetas2)+1])
taus1 = np.array(taus1[500:len(taus1)+1])
taus2 = np.array(taus2[500:len(taus2)+1])
deltaThetas = thetas1 - thetas2      


#Produce the 97.5 to generate the 95% equitable set
print(f"Bayes estimator for deltaThetas is the following: {np.mean(deltaThetas)}")
print(f"Our 95 percent equitable set is: {np.percentile(deltaThetas, [2.5,97.5])}")
print(f"The proportion of thetas1 > thetas2: {np.count_nonzero(deltaThetas>0)/len(deltaThetas)}")
print(f"Bayes estimator for thetas1 is the following: {np.mean(thetas1)}")
print(f"Bayes estimator for thetas2 is the following: {np.mean(thetas2)}")
print(f"Bayes estimator for taus1 is the following: {np.mean(taus1)}")
print(f"Bayes estimator for taus2 is the following: {np.mean(taus2)}")