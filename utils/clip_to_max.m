function X = clip_to_min(X,min_val) % clips all values to be above of equal to second lowest value.
if (nargin < 2)
    min_val = min(X(:));
end
    X(X==min_val) = min(X(X>min_val);
end