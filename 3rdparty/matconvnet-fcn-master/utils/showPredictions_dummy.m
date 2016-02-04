function [h1,h2,I1] = showPredictions_dummy(rgb,pred,scores_,labels,n,zoomBox)
nLabels = size(scores_,3);

if nargin == 6
    zoomToBox(zoomBox);
end

z = {};
z{end+1} = zeros(size2(rgb));
for t = 2:length(labels)
    z{end+1} = scores_==t-1;
end
scores_ = cat(3,z{:});

showPredictions(rgb,pred,scores_,labels,n);
