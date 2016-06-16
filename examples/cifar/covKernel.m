% Covariance metric

function covK = covKernel(res)
%This should be a local 4D array, which contains x-y-chancel-sample
%dimensions. It computes kernel wise covariance. In this version, it treats
%each points in x-y dimensions as iid.

layer_dim = size(res);

W = zeros(layer_dim(1)*layer_dim(2)*layer_dim(4),layer_dim(3));
for k=0:layer_dim(3)-1
    for x=0:layer_dim(1)-1
        for y=0:layer_dim(2)-1
            start = (x*layer_dim(2)+y)*layer_dim(4);
            W(start+1:start+layer_dim(4),k+1) = res(x+1,y+1,k+1,:);
        end
    end
end

covU = cov(W);
covK = covU;
for i=1:layer_dim(3)
    for j=1:layer_dim(3)
        covK(i,j) = covU(i,j)/sqrt(covU(i,i)*covU(j,j));
    end
end
end