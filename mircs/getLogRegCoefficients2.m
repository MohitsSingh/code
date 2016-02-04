function [ws,bs] = getLogRegCoefficients2(scores,gt_labels)
n = size(scores,2);
ws = zeros(1,n);
bs = zeros(1,n);

for k = 1:n
    X  = scores(:,k);
    [ws(k),bs(k)] = logReg(X(:), gt_labels(:));
end