########################LASSO/ADAPTIVE LASSO#####################
# Library
#install.packages("glmnet")
library(glmnet)
# Data generation
p = 20      #Number of parameters
n = 100     #Number of observations
x = rnorm(p*n,0,1)
X = matrix(x,nrow = n)
rm(x)
n0 = sample(1:p,6)
beta = rep(0,p)
beta[n0] = c(6.94, -4.03, 1.90, 3.23, 12.26, 9.99) 
beta
y = X%*%beta + rnorm(n,0,0.5)
# Lasso
lasso = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE)
lambda = lasso$lambda.min
lambda
coef.lasso = matrix(coef(lasso, s = lambda))[2:(p+1)]
lasso = glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE)
plot(lasso, xvar = "lambda", label = TRUE)
abline(v = log(lambda))
y_lasso = predict(lasso, X, s = lambda)
mse_lasso = sum((y-y_lasso)^2)/n
# Adaptive lasso
gamma = 2
b.ols = solve(t(X)%*%X)%*%t(X)%*%y
ridge = cv.glmnet(X, y, family = "gaussian", alpha = 0, intercept = FALSE)
l.ridge = ridge$lambda.min
b.ridge = matrix(coef(ridge, s = l.ridge))[2:(p+1)]
w1 = 1/abs(b.ols)^gamma
w2 = 1/abs(b.ridge)^gamma
alasso1 = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w1)
alasso2 = cv.glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w2)
lambda1 = alasso1$lambda.min
lambda2 = alasso2$lambda.min
coef.alasso1 = matrix(coef(alasso1, s = lambda1))[2:(p+1)]
coef.alasso2 = matrix(coef(alasso2, s = lambda2))[2:(p+1)]
alasso1 = glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w1)
alasso2 = glmnet(X, y, family = "gaussian", alpha = 1, intercept = FALSE, penalty.factor = w2)
plot(alasso1, xvar = "lambda", label = TRUE)
abline(v=log(lambda1))
plot(alasso2, xvar = "lambda", label = TRUE)
abline(v=log(lambda2))
View(cbind.data.frame(beta, b.ols, b.ridge, coef.lasso, coef.alasso1, coef.alasso2))
y_alasso1 = predict(alasso1, X, s = lambda1)
mse_alasso1 = sum((y-y_alasso1)^2)/n
y_alasso2 = predict(alasso2, X, s = lambda2)
mse_alasso2 = sum((y-y_alasso2)^2)/n
########################Group lasso#####################
# Library
#install.packages("gglasso")
#install.packages("fda")
#install.packages("pracma")
library(fda)
library(pracma)
library(gglasso)
# Data generation
p = 10     #Number of parameters
tp = 4     #Number of true parameters
n = 50     #Lenght of observations
m = 100    #Number of observaations
snr = 200
ds = 0.2
# Covariates
x_1 = list()
for(i in 1:p)
{
  x = seq(0,1,length=n)
  E = as.matrix(dist(x, diag=T, upper=T))
  Sigma = exp(-10*E^2)
  eig = eigen(Sigma)
  Sigma.sqrt = eig$vec%*%diag(sqrt(eig$val+10^(-10)))%*%t(eig$vec)
  mean1 = Sigma.sqrt%*%rnorm(n)
  S_noise = exp(-0.1*E^2)
  eig_noise = eigen(S_noise)
  S.sqrt_noise = eig_noise$vec%*%diag(sqrt(eig_noise$val+10^(-10)))%*%t(eig_noise$vec)
  noise = S.sqrt_noise%*%rnorm(n)
  signal = mean1 + noise
  var = var(signal)
  ds1 = sqrt(var/snr)
  S.sqrt_err = diag(n)*drop(ds1)
  x1 = matrix(0,m,n)
  for(j in 1:(m))
  {
    noise = S.sqrt_noise%*%rnorm(n)
    error = S.sqrt_err%*%rnorm(n)
    x1[j,] = mean1 + noise + error
  }
  x_1[[i]] = x1 
}
par(mfrow=c(2,5))
for(i in 1:p)
{
  matplot(t(x_1[[i]]), type = "l", xlab = i,ylab = "")
}
#Output
beta = list()
for(i in 1:tp)
{
  x = seq(0,1,length=n)
  E = as.matrix(dist(x, diag=T, upper=T))
  Sigma = exp(-1*E^2)
  eig = eigen(Sigma)
  Sigma.sqrt = eig$vec%*%diag(sqrt(eig$val+10^(-10)))%*%t(eig$vec)
  beta[[i]] =  10*Sigma.sqrt%*%rnorm(n)
}
y = rep(0,m)
for(i in 1:(m))
{
  xaux = 0
  for(j in 1:tp)
  {
    xaux = xaux + (x_1[[2*j]][i,]%*%beta[[j]])/n
  }
  y[i] = xaux + rnorm(1,0,ds)
}
dev.off()
plot(y, xlab = "")
plot(x,beta[[1]],type = "l",ylim=c(-20,20),xlab="",ylab="betas",lwd=5)
for(i in 2:tp)
{
  lines(x,beta[[i]], col = i,lwd = 5)
}
legend(0,20,c(2,4,6,8),col=c(1,2,3,4),lty = rep(1,4),lwd = 5)
# group lasso
spb = 10
splinebasis_B=create.bspline.basis(c(0,1),spb)
base_B=eval.basis(as.vector(x),splinebasis_B)
P = t(base_B)
X = array(dim=c(m,n,p))
for(i in 1:p)
{
  X[,,i] = x_1[[i]]
}
Z = array(dim=c(dim(X)[1],spb,p))
for(i in 1:p)
{
  Z[,,i] = X[,,i]%*%base_B/n 
}
Z = matrix(Z,m,spb*p)
#regression
group = rep(1:p,each=spb)
glasso = cv.gglasso(Z,y,group,loss = "ls")
lambda = glasso$lambda.min
coef = matrix(coef(glasso,s="lambda.1se")[2:(m+1)],spb,p)
View(coef)
coef = base_B%*%coef
matplot(x,coef,col=c(5,1,6,2,7,3,8,4,9,10),lty=rep(1,10),type="l",ylim=c(-20,20),lwd=5)
legend(0,20,c(1:10),col=c(5,1,6,2,7,3,8,4,9,10),lty = rep(1,10),lwd=5,)
