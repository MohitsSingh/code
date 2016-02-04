function X = clip_to_bounds(X,min_val,max_val)
    if (nargin < 2)
        min_val = 0;
    end
    if (nargin < 3)
        max_val = 1;
    end        
    X(X<min_val) = min_val;
    X(X>max_val) = max_val;
end