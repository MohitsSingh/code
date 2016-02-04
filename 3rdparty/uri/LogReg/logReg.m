function [w,b] = logReg(X, y, weights, bias)

if nargin < 2
    error('usage: [w,b] = logReg(X, y)');
end

if nargin < 3
    weights = false;
end

if nargin < 4
    bias = true;
end

[n d] = size(X);

X = [double(X), ones(n, bias)];
y = 2*double(y > 0) - 1;

options.Display = false;
if length(weights) == n
    w = minFunc(@WeightedLogisticLoss, zeros(d+bias, 1), options, X, y, weights);
else
    w = minFunc(@LogisticLoss, zeros(d+bias, 1), options, X, y);
end

if bias
    b = w(end);
    w = w(1:end-1);
end
