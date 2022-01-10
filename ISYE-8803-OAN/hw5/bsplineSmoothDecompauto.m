function [yhat,a] = bsplineSmoothDecompauto(y,B,Ba,lambda,gamma,maxIter,errtol)
%% Smooth decomposition methods
%
%
%   Parameters:
%       -----------------
%       y      :    Signal/image to be decomposed
%       B      :    Basis for mean
%       Ba     :    Basis for anomalies
%       lambda :    Smoothing Parameter, the algorithm will decide if empty
%       gamma  :    Sparsity Parameter, the algorithm will decide if empty
%       -----------------
%  Output:
%       -----------------
%       yhat   :    mean
%       a      :    Anomalies
% This code is created by Hao Yan? June 8th, 2017
% If you have any questions, please contact HaoYan@asu.edu 
% Paper: Yan, Hao, Kamran Paynabar, and Jianjun Shi. "Anomaly detection in images with smooth background via smooth-sparse decomposition." Technometrics 59.1 (2017): 102-114.

%% Test & prepare the variables
%---

plus0 = @(x) (x>0).*x;

% default parameters setup
if nargin<7
    errtol = 1e-6;
    if nargin<6
        maxIter = 20;
    end
end

isAutoLambda = isempty(lambda);
isAutoGamma = isempty(gamma);

sizey = size(y);
ndim = sum(sizey~=1);

if ndim == 1
    Lbs = 2*norm(Ba{1})^2;
    X = zeros(size(Ba{1},2),1);
    BetaA = zeros(size(Ba{1},2),1);
    
elseif ndim == 2
    Lbs = 2*norm(Ba{1})^2*norm(Ba{2})^2;
    X = zeros(size(Ba{1},2),size(Ba{2},2));
    BetaA = zeros(size(Ba{1},2),size(Ba{2},2));
end

if numel(lambda) == 1
    lambda = ones(ndim,1)*lambda;
end


SChange = 1e10;
H = cell(ndim,1);
a = zeros(size(y));
C = cell(ndim,1);
Z = cell(ndim,1);

for idim = 1:ndim
    Li = sqrtm(B{idim}'*B{idim});
    Li = Li + 1e-8*eye(size(Li));
    Di = diff(eye(size(B{idim},2)),1);
    [Ui,C{idim},~] =  svd((Li')\(Di'*Di)/(Li));
    Z{idim} = B{idim}/(Li')*Ui;
end

iIter = 0;
t = 1;

%

while SChange > errtol && iIter < maxIter
    
    iIter = iIter + 1;
    Sold = a;
    BetaSold = BetaA;
    told = t;
    gcv = @(x) splinegcv(x,y,C,Z,0,[]);
    
    if isAutoLambda && iIter==1
        lambda = fminbnd(gcv,1e-2,1e3);
        lambda = ones(ndim,1)*lambda;
    end
    % %
    
    for idim = 1:ndim
        H{idim} = Z{idim}*diag(1./(ones(size(C{idim},1),1) + lambda(idim)*diag(C{idim})))*Z{idim}'; 
    end
    if ndim == 1
        yhat = H{1}*(y-a);
        BetaSe = X + 2/Lbs*Ba{1}'*(y -Ba{1}*X - yhat);
    elseif ndim == 2
        yhat = H{1}*(y-a)*H{2};
        BetaSe = X + 2/Lbs*Ba{1}'*(y -Ba{1}*X*Ba{2}' - yhat)*Ba{2};
    end
    
    maxYe = max(abs(BetaSe(:)));
    %
    if isAutoGamma && mod(iIter,5)==1
        gamma = graythresh(abs(BetaSe)/maxYe)*maxYe*Lbs;
    end
    
    BetaA = wthresh(BetaSe,'h',gamma/Lbs); % change 'h' to 's' for softthresholding
    if ndim == 1
        a = Ba{1} *BetaA;
    elseif ndim==2
        a = Ba{1} *BetaA* Ba{2}';
    end
    t = (1+sqrt(1+4*told^2))/2;
    if iIter==1
        X = BetaA;
    else
        X = BetaA+(told-1)/t*(BetaA-BetaSold);
    end
    SChange = sum(sum(sum((a - Sold).^2)));
    
end

end

function [ GCVscore ] = splinegcv( lambda,Y,C,Z,nmiss,W )
% Estimate Generalized Cross-validation value

ndim = sum(size(Y)~=1);
H = cell(ndim,1);
dfi = zeros(ndim,1);
for idim = 1:ndim
    %%
    H{idim} = Z{idim}*diag(1./(ones(size(C{idim},1),1) + lambda*diag(C{idim})))*Z{idim}';
    dfi(idim) = sum(1./(1+lambda*diag(C{idim})));
end


df = prod(dfi);
if ndim == 1
    Yhat = H{1}*Y;
elseif ndim == 2
    Yhat = H{1}*Y*H{2};
elseif ndim >= 3
    Yhat = double(ttm(tensor(Y),H));
end

if(isempty(W))
    RSS = sum(sum(sum((Y-Yhat).^2)));
else
    RSS = sum(sum(sum((Y-Yhat).*W.*(Y-Yhat))));
end

n = numel(Y);
GCVscore = RSS/(n-nmiss)/(1-df/n)^2;
end









