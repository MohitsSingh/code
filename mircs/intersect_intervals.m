function res = intersect_intervals(x1,x2)
    n1 = size(x1,1);
    n2 = size(x2,1);
    
    inds1 = ones(size(x1));
    inds1(1:end) = 1:numel(inds1);
    inds1 = ones(size(x2));
    inds1(1:end) = 1:numel(inds1);
    
    
    found1 = false(n1,2);
    found2 = false(n2,2);
    
    
    
    
    
    
end