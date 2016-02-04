function m = mImage(imgs,displayRange)
if ~iscell(imgs)    
    m = mImage({imgs});
    return;
end
n = length(imgs);
if (nargin < 2)
    displayRange = inf;
end

if (isscalar(displayRange))
    if (isinf(displayRange))
        displayRange = [1 1 n];
    elseif (displayRange < 1)
        displayRange = [1 round(n*displayRange(1)) n];
    end
elseif (length(displayRange)==2)
    if (displayRange(2) > 1) % absolute jumps
        displayRange = [displayRange(1) displayRange(2) n];
    else % percent
        displayRange = [displayRange(1) round(n*displayRange(2)) n];
    end
    
end
displayRange(end) = min(displayRange(end),n);
displayRange = displayRange(1):displayRange(2):displayRange(3);

m = multiImage(imgs(displayRange),false,false);
% m = imresize(m,[400,NaN]);
if (nargout == 0)
    figure,imshow(m);
end
end