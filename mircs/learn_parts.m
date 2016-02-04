function [ws bs] = learn_parts(all_pos_feats,all_neg_feats,nParts,lambda)

if ~iscell(all_pos_feats)
    all_pos_feats = {{all_pos_feats}};
end
if ~iscell(all_neg_feats)
    all_neg_feats = {{all_neg_feats}};
end

if nargin < 4
    lambda = [1 .1 .01 .001];
end
zz_pos = cat(2,all_pos_feats{:});
lengths = cellfun(@length,zz_pos);
zz_pos = cat(2,zz_pos{lengths==max(lengths)});
zz_neg = cat(2,all_neg_feats{:});
lengths = cellfun(@length,zz_neg);
zz_neg = cat(2,zz_neg{lengths==max(lengths)});
N = size(zz_pos,1);
ws = {};
bs = {};
for t = 1:nParts
   r = (t-1)*N/nParts+1:t*N/nParts;
   [x,y] = featsToLabels(zz_pos(r,:),zz_neg(r,:));
   p = Pegasos(x,y,'lambda',lambda,'foldNum',5,'bias',1);
   ws{t} = p.w(1:end-1);
   bs{t} = p.w(end);
end




%[w b info] = vl_svmtrain(x, y, lambda);
