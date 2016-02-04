function [res,T] = transformBoxToImage(I_orig,M,boxFrom,toCrop,T)
if (nargin < 4 || isempty(toCrop))
    toCrop = false;
end
boxTo = [1 1 fliplr(size2(M))];
if (nargin < 5)
    T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
end
% clf; imagesc2(sal);pause
res = imtransform(M,T,'bilinear','XData',[1 size(I_orig,2)],'YData',[1 size(I_orig,1)],'XYScale',1);
if (toCrop)
    res = cropper(res,boxFrom);
end
end