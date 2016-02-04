function X = getImageStackHOG(imgs,wSize,toFlatten,zero_borders,cell_size)
if (~iscell(imgs))
    imgs = {imgs};
end
if (nargin < 2)
    wSize = [64 64];
end
if (length(wSize)>2)
    wSize = wSize(1:2);
end
if (length(wSize)==1)
    wSize = [wSize wSize];
end
if (nargin < 3)
    toFlatten = true;
end

if (nargin < 4)
    zero_borders = false;
end

if (nargin < 5)
    cell_size = 8;
end

X = cellfun2(@(x) fhog2(im2single(imResample(x,wSize,'bilinear')),cell_size),imgs);
if (zero_borders)
    for t = 1:length(X)
        x = X{t};
        x(:,1,:) = 0;
        x(1,:,:) = 0;
        x(:,end,:) = 0;
        x(end,:,:) = 0;
        X{t} = x;
%         clf;
%         subplot(1,2,1); imagesc2(imgs{t});
%         subplot(1,2,2); imagesc2(hogDraw(cat(3,x.^2,zeros(size2(x))),15,1));
%         drawnow; pause
    end
end

if (toFlatten)
    X = cellfun2(@col,X);
    X = cat(2,X{:});
end
