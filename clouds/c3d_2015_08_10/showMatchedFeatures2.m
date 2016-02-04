function hImage = showMatchedFeatures2(I1, I2, p1, p2, D, varargin)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
n1 = size(p1,1);
if isscalar(D)
    D = 1:D:n1;
end
hImage = showMatchedFeatures(I1,I2,p1(D,:),p2(D,:),varargin{:});
end

