#EXAMPLES OF FUNCTIONAL DATA ANALYSIS MODULE

########################## SPLINES ##########################

### Curve fitting ###

#Question 3
#read_csv("1,2,3\n4,5,6", col_names = c("x", "y", "z"))
library(tidyverse)
coal_df <- read_csv('C:/Users/mjpearl/Desktop/omsa/ISYE-8803-OAN/hw1/P04.csv',col_names=c("year","energy"))

#Initialize variables used throughout the question
n = length(coal_df$year)
X = seq(1,length(coal_df$year))
Y = coal_df$energy

#Data Generation
X = seq(0,1,0.001)
Y_true = sin(2*pi*X^3)^3
Y = Y_true + 0.1*rnorm(length(X))
#Define knots and basis
k = seq(0,1,length.out = 8)
k = k[2:7]
h1 = rep(1,length(X))
h2 = X
h3 = X^2
h4 = X^3
h5 = (X-k[1])^3
h5[h5 <= 0] = 0
h6 = (X-k[2])^3
h6[h6 <= 0] = 0
h7 = (X-k[3])^3
h7[h7 <= 0] = 0
h8 = (X-k[4])^3
h8[h8 <= 0] = 0
h9 = (X-k[5])^3
h9[h9 <= 0] = 0
h10 = (X-k[6])^3
h10[h10 <= 0] = 0
H = cbind(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10)
#Least square estimates
B=solve(t(H)%*%H)%*%t(H)%*%Y
plot(X,Y)   
lines(X,H%*%B,col = "red",lwd = 3)
lines(X,Y_true,col = "blue", lwd = 3)

########################## B-SPLINES ##########################

### B-splines basis ###

# Required library
library(splines)
# Basis
x = seq(1,100,length.out = 100)
par(mfrow = c(4,1))
for(sd in 1:4)
{
  knots = seq(1, 100,length.out = 10);
  B = bs(x, knots = knots, degree = sd,intercept = FALSE)
  matplot(x, B, type = "l")
}
dev.off()
image(c(1:12), x, t(B[,1:11]))

### Curve fitting ###

# Generate data:
n = 100
x = seq(0,1,length.out = n)
y = 2.5*x-sin(10*x)-exp(-10*x)
sigma = 0.3
ynoise = y + rnorm(n)*sigma
# Generate B-spline basis:
knots = seq(0,1,length.out = 10)
B = bs(x, knots = knots, degree = 2,intercept = FALSE)[,1:11]
# Least square estimation
yhat = B%*%solve(t(B)%*%B)%*%t(B)%*%ynoise
sigma2 = (1/(n-11))*t(ynoise-yhat)%*%(ynoise-yhat)
yn = yhat-3*sqrt(diag(as.numeric(sigma2)*B%*%solve(t(B)%*%B)%*%t(B)))
yp = yhat+3*sqrt(diag(as.numeric(sigma2)*B%*%solve(t(B)%*%B)%*%t(B)))
plot(x,ynoise,col = "red")
lines(x,yn,col = "blue")
lines(x,yp,col = "blue")
lines(x,yhat,col = "black")

### Fat content prediction ###
load('C:/Users/mjpearl/Desktop/omsa/ISYE-8803-OAN/hw1/meat.rdata')
# train and test data sets
s = sample.int(214,20)
train = meat[-s,]
test = meat[s,]
# model using 100 data points
linear1 = lm(X101~.,train)
pred1 = predict(linear1,test[,1:100])
rmse1 = sum((test[,101]-pred1)^2)/20
# B-splines
library(splines)
X = meat[,1:100]
x = seq(0,1,length.out =  100)
knots = seq(0,1,length.out = 8)
B = bs(x, knots = knots, degree = 3)[,1:10]
Bcoef = matrix(0,dim(X)[1],10)
for(i in 1:dim(X)[1])
{
  Bcoef[i,] = solve(t(B)%*%B)%*%t(B)%*%t(as.matrix(X[i,]))
}
meat = cbind.data.frame(Bcoef,meat[,101])
names(meat)[11] = "y"
train = meat[-s,]
test = meat[s,]
# model using 10 B-splines
linear2 = lm(y~.,train)
pred2 = predict(linear2,test[,1:10])
rmse2 = sum((test[,11]-pred2)^2)/20

########################## Smoothing splines ##########################

### Over-fitting ###

# Generate data:
n = 100
k = 40
x = seq(0,1,length.out = n)
y = 2.5*x-sin(10*x)-exp(-10*x)
sigma = 0.3
ynoise = y + rnorm(n)*sigma
# Generate B-spline basis for natural cubic spline:
   
# Least square estimation
yhat = B%*%solve(t(B)%*%B)%*%t(B)%*%ynoise
plot(x,ynoise,col = "red",lwd=3)
lines(x,yhat,col = "black",lwd=3)    

### Smoothing penalty ###

# Different lambda selection
allspar = seq(0,1,length.out = 1000)
p = length(allspar)
RSS = rep(0,p)
df = rep(0,p)
for(i in 1:p)
{
  yhat = smooth.spline(ynoise, df = k+1, spar = allspar[i])
  df[i] = yhat$df
  yhat = yhat$y
  RSS[i] = sum((yhat-ynoise)^2)
}
# GCV criterion 
GCV = (RSS/n)/((1-df/n)^2)
plot(allspar,GCV,type = "l", lwd = 3)
spar = allspar[which.min(GCV)]
points(spar,GCV[which.min(GCV)],col = "red",lwd=5)
yhat = smooth.spline(ynoise, df = k+1, spar = spar)
yhat = yhat$y
lines(x,yhat,col = "black",lwd=3) 
  
########################## Kernel smoothers ##########################

### RBF Kernel ###

# Data Genereation 
x = c(0:100)
y = sin(x/10)+(x/50)^2+0.1*rnorm(length(x)) 
kerf = function(z){exp(-z*z/2)/sqrt(2*pi)}
# leave-one-out CV
h1=seq(1,4,0.1)
er = rep(0, length(y))
mse = rep(0, length(h1))
for(j in 1:length(h1))
{
  h=h1[j]
  for(i in 1:length(y))
  {
    X1=x;
    Y1=y;
    X1=x[-i];
    Y1=y[-i];
    z=kerf((x[i]-X1)/h)
    yke=sum(z*Y1)/sum(z)
    er[i]=y[i]-yke
  }
  mse[j]=sum(er^2)
}
plot(h1,mse,type = "l")
h = h1[which.min(mse)]
points(h,mse[which.min(mse)],col = "red", lwd=5)

### Slide 8 ###

# Interpolation for N values
N=1000
xall = seq(min(x),max(x),length.out = N)
f = rep(0,N);
for(k in 1:N)
{
  z=kerf((xall[k]-x)/h)
  f[k]=sum(z*y)/sum(z);
}
ytrue = sin(xall/10)+(xall/50)^2
plot(x,y,col = "black")
lines(xall,ytrue,col = "red")
lines(xall, f, col = "blue")

################### FUNCTIONAL PRINCIPAL COMPONENTS ###################

### Functional data classification ###

library(randomForest)
# Data generation
set.seed(123)
M = 100
n = 50
x = seq(0,1,length=n)
E = as.matrix(dist(x, diag=T, upper=T))
Sigma = exp(-10*E^2)
eig = eigen(Sigma)
Sigma.sqrt = eig$vec%*%diag(sqrt(eig$val+10^(-10)))%*%t(eig$vec)
S_noise = exp(-0.1*E^2)
eig_noise = eigen(S_noise)
S.sqrt_noise = eig_noise$vec%*%diag(sqrt(eig_noise$val+10^(-10)))%*%t(eig_noise$vec)
# Class 1
mean1 = Sigma.sqrt%*%rnorm(n)
noise1 = S.sqrt_noise%*%rnorm(n)
signal1 = mean1 + noise1
var1 = var(signal1)
ds1 = sqrt(var1/100)
S.sqrt_err1 = diag(n)*drop(ds1)
x1 = matrix(0,M,n)
for(j in 1:(M))
{
  noise1 = S.sqrt_noise%*%rnorm(n)
  error1 = S.sqrt_err1%*%rnorm(n)
  x1[j,] = mean1 + noise1 + error1
}
matplot(x,t(x1),type = "l",ylab = "y",col = "blue")
# Class 2
mean2 = Sigma.sqrt%*%rnorm(n)
noise2 = S.sqrt_noise%*%rnorm(n)
signal2 = mean2 + noise2
var2 = var(signal2)
ds2 = sqrt(var2/100)
S.sqrt_err2 = diag(n)*drop(ds2)
x2 = matrix(0,M,n)
for(j in 1:(M))
{
  noise2 = S.sqrt_noise%*%rnorm(n)
  error2 = S.sqrt_err2%*%rnorm(n)
  x2[j,] = mean2 + noise2 + error2
} 
matplot(x,t(x2),type = "l",ylab = "y",col = "red")
# Train and test data sets
X = rbind(x1,x2)
lab = as.factor(c(rep(0,M),rep(1,M)))
train = sample.int(2*M,floor(0.8*2*M))
labtrain = lab[train]
labtest = lab[-train]
# Option 1: B-splines
library(splines)
knots = seq(0,1,length.out = 8)
B = bs(x, knots = knots, degree = 3)[,1:10]
Bcoef = matrix(0,dim(X)[1],10)
for(i in 1:dim(X)[1])
{
  Bcoef[i,] = solve(t(B)%*%B)%*%t(B)%*%X[i,]
}
fit = randomForest(labtrain ~ .,
                   data=cbind.data.frame(as.data.frame(Bcoef[train,]),labtrain))
pred = predict(fit,Bcoef[-train,])
table(labtest,pred)
Xtest = X[-train,]
matplot(x,t(Xtest[pred==0,]),type="l",col = "blue",ylab = "y",ylim = c(-4,4),main="Classification using B-spline coefficients")
X2 = Xtest[pred == 1,]
for(i in 1:length(pred[pred==1]))
{
  lines(x,X2[i,],col = "red")
}
# Option 2: Functional principal components
library(fda)
splinebasis = create.bspline.basis(c(0,1),10)
smooth = smooth.basis(x,t(X),splinebasis)
Xfun = smooth$fd
pca = pca.fd(Xfun, 10)
var.pca = cumsum(pca$varprop)
nharm = sum(var.pca < 0.95) + 1
pc = pca.fd(Xfun, nharm)
plot(pc$scores[lab==0,],xlab = "FPC-score 1", ylab = "FPC-score 2",col = "blue",ylim=c(-1,1))
points(pc$scores[lab==1,],col = "red")
FPCcoef = pc$scores
fit = randomForest(labtrain ~ .,
                   data=cbind.data.frame(as.data.frame(FPCcoef[train,]),labtrain))
pred = predict(fit,FPCcoef[-train,])
table(labtest,pred)
Xtest = X[-train,]
matplot(x,t(Xtest[pred==0,]),type="l",col = "blue",ylab = "y",ylim = c(-4,4),main="Classification using FPCA scores")
X2 = Xtest[pred == 1,]
for(i in 1:length(pred[pred==1]))
{
  lines(x,X2[i,],col = "red")
}

