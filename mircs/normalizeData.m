function [d,mu,stds] = normalizeData(d,mu,stds)
    if (nargin == 1)
        mu = mean(d,2);
        stds = std(d,1,2);
    end
    d = bsxfun(@minus,d,mu);
    d = bsxfun(@rdivide,d,stds);
    d(isnan(d)) = 0;
end