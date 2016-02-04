function im = standardizeImage(im,sz)
% -------------------------------------------------------------------------

% if size(im,1) > sz, im = imresize(im, [sz NaN]) ; end
im = imresize(im, [sz NaN]);