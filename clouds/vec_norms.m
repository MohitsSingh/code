function n = vec_norms(x,dim)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    dim = 2;
end
n = sum(x.^2,dim).^.5;


end

