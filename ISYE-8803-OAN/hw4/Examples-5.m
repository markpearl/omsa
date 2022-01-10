% EXAMPLES OF REGULARIZATION

%%%%%%%%%%%%%%%%%% LASSO %%%%%%%%%%%%%%%%%%

load predictors;
for i=1:100
    T(i)=X(i,:)*[0 0 0 0 0 4*ones(1,6)]'+normrnd(0,0.5);
end
[B,FitInfo] = lasso(X,T,'CV',10,'Standardize', 0 , 'Alpha' ,1 );
ax = lassoPlot(B,FitInfo, 'PlotType', 'Lambda');
B(:,FitInfo.IndexMinMSE)
FitInfo.MSE(FitInfo.IndexMinMSE)

%%%%%%%%%%%%%%%%%% ELASTIC NET %%%%%%%%%%%%%%%%%%

load predictors;
for i=1:100
    T(i)=X(i,:)*[0 0 0 0 0 4*ones(1,6)]'+normrnd(0,0.5);
end
[B1,FitInfo1] = lasso(X,T,'CV',10,'Standardize', 0 , 'Alpha' ,0.8);
ax1 = lassoPlot(B1,FitInfo1);
B1(:,FitInfo1.IndexMinMSE)
FitInfo1.MSE(FitInfo1.IndexMinMSE)
