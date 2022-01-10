
# Mohammad Salut

rm(list=ls())
library(imager)
library(pracma)
library(R.matlab)
library(cluster)    
library(factoextra)
library(profvis)
#install.packages("BiocManager")
#BiocManager::install("EBImage")
library(EBImage)

I = load.image("./Albert.jpg")
plot(I)
N = 256
hist( x = I, breaks = N, xlab = '', ylab = '',ylim= c(0,5000), main='Histogram')

### Histogram stretching ###
dev.off()
par(mfrow=c(1,3))
plot(I)
St1 <- I/(205/255)
St1[St1>1] <-  1
plot(St1)
St2 <- I/(255/150)
plot(St2,rescale=FALSE)
hist( x = I, breaks = N, xlab = '', ylab = '',ylim= c(0,5000), main='')
hist( x = St1, breaks = N, xlab = '', ylab = '',ylim= c(0,5000), main='')
hist( x = St2, breaks = N, xlab = '', ylab = '',xlim= c(0,1),ylim= c(0,5000), main='')

### IMAGE FILTTERING AND CONVOLUTION ###
rm(list=ls())
dev.off()
Y <- readImage("Albert.jpg")
K <- array(0, dim = c(3,3, 6))
K[,,1] <- matrix(c(1,0,-1, 0,0,0, -1,0,1), nrow = 3, ncol = 3)  #Edge Detection
K[,,2] <- matrix(c(0,1,0,  1,-4,1, 0,1,0), nrow = 3, ncol = 3) #Edge Detection
K[,,3] <- matrix(c(-1,-1,-1, -1,8,-1, -1,-1,-1), nrow = 3, ncol = 3) #Edge Detection
K[,,4] <- matrix(c(0,-1,0, -1,5,-1, 0,-1,0), nrow = 3, ncol = 3) #Sharpening 
K[,,5] <- matrix(1/9, nrow = 3, ncol = 3) #Blurring
K[,,6] <- matrix(c(1,2,1, 2,4,2, 1,2,1)/16, nrow = 3, ncol = 3) #Blurring
par(mfrow=c(2,3))
for (i in 1:6){
  Yk <- filter2(Y,K[,,i])
  plot(Yk)
}

### Denoising using splines ###
### 2D example: Generate data
rm(list=ls())
dev.off()
peaks <- function(n){
  t <- seq(-3,3,length.out=n)
  x <- meshgrid(t, t)$X
  y <- meshgrid(t, t)$Y
  Z <-  3*(1-x)^2*exp(-(x^2) - (y+1)^2) - 10*(x/5 - x^3 - y^5)*exp(-x^2-y^2)- 1/3*exp(-(x+1)^2 - y^2)
  return(Z)
}
n <- 100
Y <- peaks(n) + 0.5*matrix(rnorm(n*n),n,n)
par(mfrow=c(1,2))
image(Y)
#2D Spline
k <- 10
x=seq(1,100,length.out=100)
knots = seq(1, 100,length.out = 8)
library(splines)
B = bs(x, knots = knots, degree = 3,intercept = FALSE)[,1:10]
H = B%*%solve(t(B)%*%B)%*%t(B)
Yhat = H%*%Y%*%H
image(Yhat)

#######IMAGE SEGMENTATION######
###### Otsu's Method 
###### Step 1: get the histogram of image
rm(list=ls())
dev.off()
I_grey = load.image("coins.jpg")
N=255
counts = hist( x = I_grey, breaks = c(0:N)/N, xlab = '', ylab = '', main='Histogram of Grayscale Image')
###Step 2: Calculate group mean
p = counts$counts/ sum(counts$counts)
omega = cumsum(p)
mu = cumsum(p * t(1:N))
mu_t = tail(mu, 1)
### Step 3: find the maximum value of 
sigma_b_squared = (mu_t * omega - mu)^2 / (omega * (1 - omega))
sigma_b_squared[is.nan(sigma_b_squared)] <- 0
maxval = max(sigma_b_squared)
idx = mean(which(sigma_b_squared == maxval))
### Step 4: Thresholding and get final image
level = (idx - 1) / (N - 1)
BW = I_grey
BW[I_grey>level] = 1
BW[I_grey<=level] = 0
plot(BW)
########## K-means clustering ###################### 
#Load Data
rm(list=ls())
dev.off()
library(OpenImageR)  
I <- readImage("coin.jpg")
imageShow(I)
X <- Reshape(I, dim(I)[1]*dim(I)[2],dim(I)[3])
K <- 2
max_iter <- 10
# Clustering
N <- dim(X)[1]
d <- dim(X)[2]
# indicator matrix (each entry corresponds to the cluster of each point in X)
L <- rep(0,N)
# centers matrix
C <- Reshape(rep(0,K* d),K,d)
for (i in 1:max_iter){
# step 1: optimize the labels
  dist <- Reshape(rep(0,N* K),N,K)
  for (j in 1:N){
    for(k in 1:K){
      dist[j,k] <- norm(matrix(X[j,]-C[k,]), type = c("F"))
    }
  }
  disto <- matrix(rep(0,dim(dist)[1]*dim(dist)[2]),dim(dist)[1],dim(dist)[2])
  index <- disto
  disto=t(apply(dist,1,sort))
  index=t(apply(dist,1,order))
  L <- index[,1]
 # step 2: optimize the centers
  for (k in 1:K){
    if (sum(L==k) !=0){
      for (ij in 1:dim(X[L==k,])[2]){
        C[k,ij]=sum(X[L==k,ij])/sum(L==k)
      }
    }
  }
}
Y <- Reshape(L, dim(I)[1],dim(I)[2])
BW <- Y == 1
plot(as.raster(BW))
### k-means clustering ###
### input image
dev.off()
I <- readImage("CS.jpg")
imageShow(I)
dm <- dim(I)

rgbImage <- data.frame(
  x=rep(1:dm[2], each=dm[1]),
  y=rep(dm[1]:1, dm[2]),
  r.value=as.vector(I[,,1]),
  g.value=as.vector(I[,,2]),
  b.value=as.vector(I[,,3]))

plot(y ~ x, data=rgbImage,  col = rgb(rgbImage[c("r.value", "g.value", "b.value")]),  asp = 1, pch = ".")

kColors <- c(2,3,4,5) 

for (i in 1:4){
  kMeans <- kmeans(rgbImage[, c("r.value", "g.value", "b.value")],  centers = kColors[i],iter.max = 5000)
  clusterColour <- rgb(kMeans$centers[kMeans$cluster, ])
  plot(y ~ x, data=rgbImage, col = clusterColour, axes=FALSE,  asp = 1, pch = ".")
  pause(5)
  dev.off()
}

########## EDGE DETECTION #############
  
###### Sobel operator ######
library(OpenImageR)  
rm(list=ls())
dev.off()
a <- readImage("C:/Users/mjpearl/Desktop/omsa/ISYE-8803-OAN/hw2/ExampleCodeR-6-1/coins.jpg")
op <- matrix(c(1,0,-1, 2,0,-2, 1,0,-1), nrow = 3, ncol = 3)/8
x_mask <- t(op)
y_mask <- op
fx <- filter2(a, x_mask, boundary = c("circular", "replicate"))
fy <- filter2(a, y_mask, boundary = c("circular", "replicate"))
imageShow(imageData(fx))
imageShow(imageData(fy))
f = fx * fx + fy * fy
imageShow(imageData(f))
# The edge_detection function uses one of the Frei_chen, LoG (Laplacian of Gaussian), Prewitt, Roberts_cross, Scharr or Sobel filters to perform edge detection
f = edge_detection(a@.Data, method = 'Sobel', conv_mode = 'same')
plot(as.raster(f))


