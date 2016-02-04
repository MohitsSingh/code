function [pred,scores,t] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,predVar,rgb,n,toShow,labels)
if nargin < 7
    toShow=true;
end
if nargin < 8
    labels = [];
end
tic
[scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar,predVar);
t=toc;
if toShow        
    softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
    showPredictions(rgb,pred,softmaxScores,labels,n)
end


