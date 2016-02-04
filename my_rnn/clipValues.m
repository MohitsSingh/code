function V = clipValues( V,minVal,maxVal )
%CLIPVALUES Summary of this function goes here
%   Detailed explanation goes here
V(V<minVal) = minVal;
V(V>maxVal) = maxVal;

end

