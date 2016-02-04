function im = imreadx(ex)
% Read a training example image.
%   im = imreadx(ex)
%
% Return value
%   im    The image specified by the example ex
%
% Argument
%   ex    An example returned by pascal_data.m
if (~ischar(ex.im))
    im = ex.im;
else
    im = imread(ex.im);
end
im = color(im);
if ex.flip
  im = im(:,end:-1:1,:);
end
