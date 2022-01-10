function hat_y=omp(s,T,N)

Size=size(T);                                     %  Size of measuremetn Matrix
M=Size(1);                                        %  Measrue
hat_y=zeros(1,N);                                 %  coefficiet to be recovered
Aug_t=[];                                         %  Augmentaion matrix
r_n=s;                                            %  error

for times=1:M;                                    %  Iteration number

    for col=1:N;                                  %  recover all columns
        product(col)=abs(T(:,col)'*r_n);          %  Recover inner product
    end
    [val,pos]=max(product);                       %  Maximum inner product
    Aug_t=[Aug_t,T(:,pos)];                       %  augment matrix
    T(:,pos)=zeros(M,1);                          %  zero pixed column
    aug_y=(Aug_t'*Aug_t)^(-1)*Aug_t'*s;           %  Least squre
    r_n=s-Aug_t*aug_y;                            %  Residual
    pos_array(times)=pos;                         %  Find residual largest point
    
    if (abs(aug_y(end))^2/norm(aug_y)<0.05)       %  Find best error cut off
        break;
    end
end

hat_y(pos_array)=aug_y;                           %  Recover