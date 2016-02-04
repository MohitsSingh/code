function q = blendRegion(im,region,alpha_,color_)
if (length(size(im))<3)
    im = cat(3,im,im,im);
end
if (nargin  < 3)
    alpha_ = .3;
end
if (nargin < 4)
    color_ = [1 0 0];
end
if (length(size(im)) == 2)
    im = repmat(im,[1 1 3]);
end
if (alpha_==-1)
    q = im;
else
    region = padarray(region,max(0,dsize(im,1:2)-size(region)),0,'post');
    q = im.*((1-alpha_)+alpha_*repmat(region,[1 1 3]));
end
% B = imdilate(bwperim(region),ones(3));
B = bwperim(region);
% B = imdilate(B,ones(3));
qR = q(:,:,1);
qG = q(:,:,2);
qB = q(:,:,3);
qR(B) = color_(1);
qG(B) = color_(2);
qB(B) = color_(3);
q = cat(3,qR,qG,qB);