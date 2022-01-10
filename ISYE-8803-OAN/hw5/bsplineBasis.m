function [ B,Bd] = bsplineBasis(n,k,sd)
% bsplineBasis: Construct k Bspline Basis with n gridded with spline degree sd
% Input Variable: 
% n: length of signal or number of pixels 
% k: number of knots, k = n, sd must be zero, then I matrix
% sd: spline degree, sd = 0, then constant function
% bd: how many basis in boundary. 
% This code is created by Hao Yan? June 8th, 2017
% If you have any questions, please contact HaoYan@asu.edu 
% Paper: Yan, Hao, Kamran Paynabar, and Jianjun Shi. "Anomaly detection in images with smooth background via smooth-sparse decomposition." Technometrics 59.1 (2017): 102-114.

if n == k 
    B = eye(n);
else
    bd = sd-1;
    knots = [ones(1,bd) linspace(1,n,k) n * ones(1,bd)];
    nKnots = length(knots) - sd;
    kspline = spmak(knots,eye(nKnots));
    kd = fnder(kspline);
    B=spval(kspline,1:n)';
    Bd = spval(kd,1:n)';    
end


end

