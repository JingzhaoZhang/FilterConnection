function [V,D] = LaplacianEigMap(W, weightMargin)
% paper: Laplacian Eigenmaps for Dimensional Reduction and Data
% Representation, Belkin & Niyogi 03'

dimW = size(W);
W = W - diag(diag(W));
for i= 1:dimW(1)
    for j=1:dimW(1)
        W(i,j) = max(W(i,j),-W(i,j));
    end
end
W(W<weightMargin) = 0;
D = diag(sum(W,1));
L = D - W;
% Solve the generalized eignevalue problem
D_inv2 = sqrt(1./diag(D));
M = L .* (D_inv2 * D_inv2');
[V,D] = eig(M);
V = diag(D_inv2) * V;
end