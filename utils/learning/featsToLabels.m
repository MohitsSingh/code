function [x,y] = featsToLabels(pos_feats,neg_feats)
x = [pos_feats,neg_feats];
y = zeros(size(x,2),1);
y(1:size(pos_feats,2)) = 1;
y(size(pos_feats,2)+1:end) = -1;