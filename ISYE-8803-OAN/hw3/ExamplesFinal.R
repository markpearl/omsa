
# Mohammad Salut

##  EXAMPLES OF TENSOR ANALYSIS MODULE
#### TENSOR PRELIMNARIES I ####
### Inner product ###
library(rTensor)
library(R.matlab)

x <-  rnorm(24, mean=0, sd=1)
dim(x) <- c(4,2,3)
X <- as.tensor(x)
X@data

y <-  rnorm(24, mean=0, sd=1)
dim(y) <- c(4,2,3)
Y <- as.tensor(y)
Y@data
Z <- X * Y
Z@data

### Matricization ###
X1 <- matrix(c(1,2,3,4), nrow = 2, ncol = 2)
X2 <- matrix(c(5,6,7,8), nrow = 2, ncol = 2)
m = rep(1,8)
dim(m) = c(2,2,2)
X = as.tensor(m)
X[,,1]=X1
X[,,2]=X2
X@data
A <- k_unfold(X,1) #<-- Mode-1 matricization
A@data
B <- k_unfold(X,2) #<-- Mode-2 matricization
B@data
C <- k_unfold(X,3) #<-- Mode-3 matricization
C@data
######## TENSOR PRELIMNARIES II ###########
##### n-mode matrix product using rTensor ###
X <- rand_tensor(modes = c(5, 3, 4), drop = FALSE) #Generate a Tensor with specified modes with iid normal(0,1) entries
X@data
A <-  matrix(rnorm(20, mean=0, sd=1),nrow = 4, ncol = 5)
Y <- ttm(X, A, m = 1) #<-- X times A in mode-1
Y@data
#### Kronecker product ###
A <- matrix(c(1,3,5,2,4,6), nrow = 3, ncol = 2)
B <- matrix(c(1,4,2,5,3,6), nrow = 2, ncol = 3)
C <- kronecker(A,B)
C
### Khatri-Rao Poduct ####
A <- matrix(c(1,3,5,2,4,6), nrow = 3, ncol = 2)
B <- matrix(c(1,3,2,4), nrow = 2, ncol = 2)
C <- khatri_rao(A,B)

### Hadamard Product ###
A <- matrix(c(1,3,5,2,4,6), nrow = 3, ncol = 2)
B <- matrix(c(1,3,5,2,4,6), nrow = 3, ncol = 2)
C <- A * B
#################### TENSOR DECOMPOSITION I ##############
##### Cp decomposition #####
x <-  rep(0, 60)
index <- sample(1:60, 10)
randomnom <- runif(10)
x[index] <- randomnom
dim(x) <- c(5,4,3)
X <- as.tensor(x)
P <- cp(X, num_components = 2, max_iter = 50, tol = 1e-05)
P$conv   #in order to ensure convergence, as can be checked with the conv attribute
P$lambdas
P$U

### Heat transfer example ###
rm(list = ls())
x <- readMat('X.mat')$X
par(mfrow=c(2,5))
for (t in 1:10){
  image(x[,,t])
}
X <- as.tensor(x) ## read the MATLAB generated Heat transfer data
### CP Decomposition
## min(IJ,JK,IK) = 21*10 = 210
err <- rep(0,10)
AIC <- rep(0,10)
for (i in 1:10){
  P <- cp(X, num_components = i, max_iter = 50, tol = 1e-05)
  DD <- as.tensor(P$est@data)
  err[i] <- fnorm(X - DD)^2
  AIC[i] <- 2*err[i] + 2*i
}
Min <- which.min(AIC)
Min
P <- cp(X, num_components = Min , max_iter = 50, tol = 1e-05)
# We need to sort lamdas in descending order
lambdas <- P$lambdas
index <- order(lambdas, decreasing = TRUE)
lambdas <- lambdas[index]
A <- P$U[[1]][,index]
B <- P$U[[2]][,index]
C <- P$U[[3]][,index]
dev.off()
matplot(C,type="l",main="",xlab="",ylab="")
par(mfrow=c(1,Min))
for (i in 1:Min){
  XY <- kronecker(A[,i],t(B[,i]))*lambdas[i]
  image(XY)
}

###  Tucker decomposition ####
rm(list = ls())
x <-  rep(0, 60)
index <- sample(1:60, 10)
randomnom <- runif(10)
x[index] <- randomnom
dim(x) <- c(5,4,3)
X <- as.tensor(x)
T<- tucker(X, ranks =c(2,2,1), max_iter = 25, tol = 1e-05)
(T$Z)@data 
T$U
###### Heat transfer example ####
###### Data generation is the same 
###### Tucker Decomposition
rm(list = ls())
x <- readMat('C:/Users/mjpearl/Desktop/omsa/ISYE-8803-OAN/hw3/X.mat')$X
X <- as.tensor(x) 
X1 <- k_unfold(X,1)
err <- rep(0,64)
dim(err) <- c(4,4,4)
AIC <- rep(0,64)
dim(AIC) <- c(4,4,4)
for(i in 1:4){
  for(j in 1:4){
    for(k in 1:4){
      T<- tucker(X, ranks =c(i,j,k), max_iter = 50, tol = 1e-05)
      DD <- as.tensor(T$est@data)
      err[i,j,k] <- fnorm(X - DD)^2
      AIC[i,j,k] <- 2*err[i,j,k] + 2*(i+j+k)
    }
  }
}
AIC
min(AIC)
T<- tucker(X, ranks =c(3,3,3), max_iter = 50, tol = 1e-07)
A <- T$U[[1]]
B <- T$U[[2]]
C <- T$U[[3]]
XY1 <- kronecker(A[,1],t(B[,1]))
XY2 <- kronecker(A[,2],t(B[,2]))
XY3 <- kronecker(A[,3],t(B[,3]))
dev.off()
par(mfrow=c(2,2))
par(mar = rep(2, 4))
image(XY1)
image(XY2)
image(XY3)
matplot(C,type="l",main="",xlab="",ylab="")
