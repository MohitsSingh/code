function D = batchDistance(conf,C,ids,D)
%BATCHDISTANCE computes distance of nearest neighbor for each feature in "ids"
% to each feature in x
if (nargin < 4)
    D = zeros(length(ids),size(C,2));
end
for k = 1:length(ids)
    k
    if (any(D(k,:)==0))
        curImage = getImage(conf,ids{k});
        
        X =allFeatures(conf,curImage);
        if (numel(X)>0)
            d = l2(X',C');
            [d_min] = min(d,[],1);
            D(k,:) = d_min;
        else
            D(k,:) = inf;
        end
    end
end

end

