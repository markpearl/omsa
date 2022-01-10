% EXAMPLES OF REGULARIZATION APLICATIONS

%%%%%%%%%%%%%%%%%% Compressive Sensing II %%%%%%%%%%%%%%%%%%

%%% Example 1: Sparsity in time domain %%%
% Generate signal
T = 0 : 1/15e9 : 30e-9;
D = [15e-9 ; 1]';
x = pulstran(T,D,@gauspuls,4E9,.5);
figure;
plot(T,x)
% FFT analysis
xf = fft(x);     
figure;
plot(abs(xf))
% Compressive sensing
N = size(T,2);
K = 90;
B = dftmtx(N); %creating Discrete Fourier matrix
q = randperm(N); %selecting random rows of the DFT matrix
A = B(q(1:K),:); %creating measurement matrix
y = (A*x'); %taking random frequency measurements
x0 = A'*y; %Calculating Initial guess
xp = l1eq_pd_matlab(x0,A,[],y,1e-5); %Running the recovery Algorithm
figure;
plot(T,xp)

%%% Example 2: Sparsity in frequency domain %%%
% Generate signal
N = 1024;
n = (0:(N-1))';
k1 = 30;
k2 = 80;
k3 = 100;
x=(sin(2*pi*(k1/N)*n)+sin(2*pi*(k2/N)*n)+sin(2*pi*(k3/N)*n))';
figure;
plot(x)
% FFT analysis
xf = fft(x);     
figure;
plot(abs(xf))
% Compressive sensing
K = 650;
ID = eye(N);
q = randperm(N);
Phi = ID(:,q(1:K))';
Psi = dftmtx(N);
xf = Psi*x';
y = (Phi*x'); %taking random time measurements
x0 = Psi'*(Phi'*y); %Calculating Initial guess
xp = l1eq_pd_matlab(x0,Phi*Psi,[],y,1e-7); %Running the recovery Algorithm
xprec=real(-inv(Psi)*xp); %recovered signal in time domain
figure;
plot(xprec)

%%% Example 3: CS Application for images %%%
% See folder 2D wavelet

%%% Example 4: Noisy signal recovery %%%
% See folder 1D Noise

%%%%%%%%%%%%%%%%%% Matrix Completion %%%%%%%%%%%%%%%%%%

%%% Example %%%
%Generating original matrix
n1 = 10; n2 = 8;
A = randi([-20,20],n1,n2);
r = 2;
[U, S, V] = svd(A);
if n1 < n2
    s = diag(S); s(r+1:end)=0; S=[diag(s) zeros(n1,n2-n1)];
else
    s = diag(S); s(r+1:end)=0; S=[diag(s); zeros(n1-n2,n2)];
end
X = U* S* V';
X0 = X;
%Removing 20% of the observations
A = [rand(n1,n2)>=0.80];
X(A) = 0;
m = sum(sum(A==0));
%Initialization
Y=zeros(n1,n2);
delta = n1*n2/m;
tau = 250;
%Iterations
vec = zeros(500,1);
for i = 1:500
    [U, S, V] = svd(Y);
    S_t = (S-tau);
    S_t(S_t<0) = 0;
    Z = U*S_t*V';
    P = X-Z;
    P(A) = 0;
    Y0 = Y;
    Y = Y0 + delta*P;
    vec(i) = sum(sum((Y-Y0).^2));
    err(i)=sum(sum((X0-Z).^2))/sum(sum((X0).^2));
end
% plot the results
figure;plot(vec);
figure;plot((err));
figure;
Ar=reshape(A, n1*n2,1);
Xr=reshape(X0, n1*n2,1);Xr=Xr(Ar);
Zr=reshape(Z, n1*n2,1);Zr=Zr(Ar);
subplot(2,1,1);plot(Xr);hold on;plot(Zr,'r');
subplot(2,1,2);plot(Xr-Zr);
figure;
imagesc(Z)
figure;
imagesc(X0)

%%%%%%%%%%%%%%%%%% Robust PCA %%%%%%%%%%%%%%%%%%

%%% Example %%%
D = double(rgb2gray(imread('building.png')));
figure;
imshow(uint8(D))
lambda=1e-2;
[m n] = size(D); tol = 1e-7; maxIter = 1000;
% Initialize A,E,Y,u
Y = D; norm_two = norm(Y); norm_inf = norm( Y(:), inf) / lambda;
Y = Y / norm_inf;
A_hat = zeros( m, n); E_hat = zeros( m, n);
mu = 1.25/norm_two; mu_bar = mu * 1e7; rho = 1.5;         % this one can be tuned
d_norm = norm(D, 'fro');
iter = 0; total_svd = 0; converged = false; stopCriterion = 1;
while ~converged
    iter = iter + 1;
    temp_T = D - A_hat + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat+min(temp_T + lambda/mu, 0);
    [U S V] = svd(D - E_hat + (1/mu)*Y, 'econ');
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    A_hat = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';
    total_svd = total_svd + 1;
    Z = D - A_hat - E_hat;
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
    % stop Criterion
    stopCriterion = norm(Z, 'fro') / d_norm;
    if stopCriterion < tol
        converged = true;
    end
end
figure;
imshow(uint8(A_hat))
figure;
imshow(uint8(E_hat))

%%%%%%%%%%%%%%%%%% Smooth Sparse Decomposition %%%%%%%%%%%%%%%%%%

%%% Example %%%
% This code is created by Hao Yan? June 8th, 2017
% If you have any questions, please contact HaoYan@asu.edu
% Paper: Yan, Hao, Kamran Paynabar, and Jianjun Shi. "Anomaly detection in images with smooth background via smooth-sparse decomposition." Technometrics 59.1 (2017): 102-114.
load data/data.mat
sigma = 0.05;
delta = 0.2;
Y = Y0 + delta*A0 + normrnd(0,sigma,size(A0,1),size(A0,2));
figure;imagesc(Y)
kx = 6; ky = 6;
nx = size(Y,1); ny = size(Y,2);
B{1} = bsplineBasis(nx,kx,3);
B{2} = bsplineBasis(ny,ky,3);
sd = 3;
snk = 4;  skx = round(nx/snk); sky = round(ny/snk);
Bs{1} = bsplineBasis(nx,skx,2);
Bs{2} = bsplineBasis(ny,sky,2);
[yhat,a] = bsplineSmoothDecompauto(Y,B,Bs,[],[]);
figure
colormap('jet')
subplot(1,2,1)
imagesc(yhat)
title('Mean')
set(gca,'FontSize',14)
subplot(1,2,2)
imagesc(a)
title('Anomalies')
set(gca,'FontSize',14)

%%%%%%%%%%%%%%%%%% Kernel Ridge Regression %%%%%%%%%%%%%%%%%%

%%% Example %%%
clc
Xtrain = (1:100)/100;
Yt = sin(Xtrain*10)+(Xtrain*2).^2;
Ytrain = Yt + 0.2*randn(1,100);
N = 100; 
Xtest = linspace(min(Xtrain),max(Xtrain),N);
Xtrain=Xtrain(:); Ytrain=Ytrain(:); n = length(Xtrain);Xtest=Xtest(:); 
lambda = 0.04;
c = 0.04; 
kernel1 = exp(-dist(Xtrain').^2 ./ (2*c));
kernel2 = exp(-pdist2(Xtrain, Xtest).^2 ./ (2*c));
yhatRBF = Ytrain' * ((kernel1 + lambda * eye(size(kernel1))) \ kernel2);
plot(Yt, '-b')
hold on
plot(Ytrain, 'bo') 
hold on
plot(yhatRBF, '-- r')
hold off
