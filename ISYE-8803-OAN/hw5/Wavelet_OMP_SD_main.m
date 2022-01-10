
clc;clear

%%  CS Measurement
% readfile
X=imread('lena256.bmp');
X=double(X);
[a,b]=size(X);

%  Measurement matrix
M=190;
R=randn(M,a);

% Measurement in Original domain
Y=R*X;

%%  CS Recover

%  Wavlet matrix
ww=DWT(a);
%  Measure value
Y=Y*ww';
%  Measure Matrix
R=R*ww';
%%
%  OMP algorithm
X2=zeros(a,b);  %  Recover matrix
for i=1:b  % column permulation
    rec=omp(Y(:,i),R,a);
    X2(:,i)=rec;
end

%%  CS result

% original Image
figure(1);
imshow(uint8(X));
title('original Image');

% Transfered Image
figure(2);
imshow(uint8(X2));
title('Transfered Image');

% Recovered image
figure(3);
X3=ww'*sparse(X2)*ww;  %  inverse DWT
X3=full(X3);
imshow(uint8(X3));
title('Recovered Image');

% Error
errorx=sum(sum(abs(X3-X).^2));        %  MSE
psnr=10*log10(255*255/(errorx/a/b))   %  PSNR
