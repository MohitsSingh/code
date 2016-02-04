function [src,dst] = disparity_to_src_dst(S)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [X,Y] = meshgrid(1:size(S,2),1:size(S,1));
    X_dst = X+S;
    
    m = ~isnan(S(:));
    src = double([X(m) Y(m)]);
    dst = double([X_dst(m) Y(m)]);

end

