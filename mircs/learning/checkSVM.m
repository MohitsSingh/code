function [w b ap] = checkSVM( X,train_labels )
%CHECKSVM Summary of this function goes here
%   Detailed explanation goes here
X(isinf(X(:))) = min(X(~isinf(X(:))));
% %
% % % X = vl_homkermap(X', 1, 'KJS')';
% %
y = double(train_labels);
% % % %
foldNum = 2;
[n d] = size(X);
cvIdx = crossvalind('Kfold', n, foldNum);
testIdx = cvIdx == 1;
trainIdx = ~testIdx;
% 
% [w b] = vl_pegasos(X(trainIdx, :)', int8(2*(y(trainIdx) > 0 ) - 1)', 1e-10, 'biasMultiplier', 1, 'iterations', 1e5);
[w b info] = vl_pegasos(X(trainIdx, :)', int8(2*(y(trainIdx) > 0 ) - 1)',...
.00001,'biasMultiplier',1,'maxiterations',500);
% w
% b
s = X(testIdx, :)*w;%(1:end-1) + w(end);
[p r ap] = calc_aps2(s, y(testIdx)); disp(ap);

end

