function [w b] = concat_and_learn(all_pos_feats,all_neg_feats,lambda)
if ~iscell(all_pos_feats)
    all_pos_feats = {{all_pos_feats}};
end
if ~iscell(all_neg_feats)
    all_neg_feats = {{all_neg_feats}};
end

if nargin < 3
    lambda = [.1 .01 .001 .0001];
end
zz_pos = cat(2,all_pos_feats{:});
lengths = cellfun(@length,zz_pos);
zz_pos = cat(2,zz_pos{lengths==max(lengths)});
zz_neg = cat(2,all_neg_feats{:});
lengths = cellfun(@length,zz_neg);
zz_neg = cat(2,zz_neg{:});
[x,y] = featsToLabels(zz_pos,zz_neg);

%[w b info] = vl_svmtrain(x, y, lambda);
p = Pegasos(x,y,'lambda',lambda,'foldNum',5,'bias',1);
w = p.w(1:end-1);
b = p.w(end);