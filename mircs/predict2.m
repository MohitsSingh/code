function p = predict2(feats,w)    
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if isstruct(w)
        w = cat(2,w.w);
    end
    r = (w(:,1:end-1)*feats);
    p = bsxfun(@plus,r,w(:,end))';
end

