
% Get Wavelet Transofrmation Matrix with Image size N*N, and N=2^P

function ww=DWT(N)

[h,g]= wfilters('sym8','d');       %  separate low pass and high pass

% N=256;                           %   Size of Matrix
L=length(h);                       %  Length of bandwidth
rank_max=log2(N);                  %  Maximum Layer
rank_min=double(int8(log2(L)))+1;  %  Minimum Layes
ww=1;                              % Proprocessing Matrix

% Matrix construction
for jj=rank_min:rank_max
    
    nn=2^jj;
    
    % Construct Vectorè
    p1_0=sparse([h,zeros(1,nn-L)]);
    p2_0=sparse([g,zeros(1,nn-L)]);
    
    % circlular move
    for ii=1:nn/2
        p1(ii,:)=circshift(p1_0',2*(ii-1))';
        p2(ii,:)=circshift(p2_0',2*(ii-1))';
    end
    
    % Orthogonal Matrix
    w1=[p1;p2];
    mm=2^rank_max-length(w1);
    w=sparse([w1,zeros(length(w1),mm);zeros(mm,length(w1)),eye(mm,mm)]);
    ww=ww*w;
    
    clear p1;clear p2;
end
