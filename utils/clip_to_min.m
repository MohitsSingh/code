function X = clip_to_min(X) % clips all values to be above of equal to second lowest value.    
    X(X==min(X(:))) = min(X(X>min(X(:))));    
end