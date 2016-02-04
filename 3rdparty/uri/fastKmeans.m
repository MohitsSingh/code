function [clust cents] = fastKmeans(X, k, varargin)

args = Args(varargin);
epsi = args.get('epsi', 1e-3);
maxIter = args.get('maxIter', 100);
verbose = args.get('verbose', false);

[n d] = size(X);
sampIdx = col(1:n);

cents = zeros(k, d);
clust = randi(k, n, 1);
loop = true;
i = 1;
while loop && (i <= maxIter)
    i
    prev = cents;
    cents(unique(clust), :) = cell2mat(accumarray(clust, sampIdx, [k, 1], @(idx) {mean(X(idx, :), 1)}));        
    clust = argmin(l2(X, cents), 2);    
    
    diff = max(abs(col(prev - cents)));
    loop = diff > epsi;

    if verbose
        disp([i diff]);
    end
    i = i + 1;    
end
