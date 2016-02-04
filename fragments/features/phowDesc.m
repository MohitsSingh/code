function [f,d] = phowDesc(m,varargin)
%TINYDESC Summary of this function goes here
if (length(size(m))==3)
    m = rgb2gray(m);
end
m = im2single(m);
[f,d] = vl_phow(m,varargin{:});
% d = rootSift(d);

end