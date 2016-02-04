function [feats,labels,valids] = balanceData(feats,labels,direction,valids)
if nargin < 4
    valids = true(1,length(labels));
end
if (direction == 0)
    return
end
    
nPos = nnz(labels==1);
nNeg = length(labels)-nPos;

feats_pos = feats(:,labels==1);
feats_neg = feats(:,labels~=1);

valids_pos = valids(labels==1);
valids_neg = valids(labels~=1);

if (direction == 1)
    feats_pos = repmat(feats_pos,1,max(1,round(nNeg/nPos)));
elseif direction ~=0
    sel_ = 1:size(feats_neg,2);
    sel_ = vl_colsubset(sel_,nPos,'uniform');
    feats_neg = feats_neg(:,sel_);
    valids_neg = valids_neg(sel_);

end

feats = [feats_pos,feats_neg];
valids = [valids_pos(:);valids_neg(:)];

labels = [ones(size(feats_pos,2),1);-ones(size(feats_neg,2),1)];
end