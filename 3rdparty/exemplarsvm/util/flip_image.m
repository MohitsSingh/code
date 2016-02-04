function I = flip_image(I,dim)
%Flips the image in the LR direction
if (nargin < 2)
    dim = 2;
end
for i = 1:size(I,3)
    if (dim==2)
        I(:,:,i) = fliplr(I(:,:,i));
    else
        I(:,:,i) = flipud(I(:,:,i));
    end
end