%%% Example %%%
% This code is created by Hao Yan? June 8th, 2017
% If you have any questions, please contact HaoYan@asu.edu
% Paper: Yan, Hao, Kamran Paynabar, and Jianjun Shi. "Anomaly detection in images with smooth background via smooth-sparse decomposition." Technometrics 59.1 (2017): 102-114.
load data/data.mat
sigma = 0.05;
delta = 0.2;
Basis{1} = B; % convert the given B into a cell
Basisa{1} = Ba; % convert the given Ba into a cell
[yhat,a] = bsplineSmoothDecompauto(Y,Basis,Basisa,[],[]);
figure
subplot(1,3,1)
plot(yhat)
title('Smooth')
set(gca,'FontSize',14)
subplot(1,3,2)
plot(a)
title('Anomalies')
set(gca,'FontSize',14)
subplot(1,3,3)
plot(Y-yhat-a)
title('Error')
set(gca,'FontSize',14)
