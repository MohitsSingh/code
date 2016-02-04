function res = intersect_intervals2(x)
n = size(x,1);
inds = repmat(col(1:n),1,2);

opens = false(n,1);
res = false(n);

[~,ir] = sort(x(:));
inds = inds(ir)';
numOpen = 0;




for curInd = inds
    if (opens(curInd))
        opens(curInd) = 0;
        numOpen = numOpen-1;
    else
        numOpen = numOpen+1;
        % this must be the opening part
        % add intersection between cur index and all previous
        res(opens,curInd) = 1;
        opens(curInd) = 1;
    end
    % there's an intersection between all previous open
    
end