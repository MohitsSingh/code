
function m = showSorted(images,scores,n)
% SHOWSORTED display some images sorted by ascending score.
% SHOWSORTED(images,scores,n) : shows an array of images, given in a cell
% array, sorted by descending values of scores, up to a maximum of n
% images. If n is not given, all images are shown. 
% m = SHOWSORTED(...) does not display the result but return it into an
% image.
if (nargin < 3)
    n = length(images);
else
    if (isscalar(n))
        n = min(n,length(images));
    end
end

[r,ir] = sort(scores,'descend');
m = [];
if (nargout == 0)
    mImage(images(ir(1:n)));
else
    m = mImage(images(ir(1:n)));
end


end