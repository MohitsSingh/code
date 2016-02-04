function displayImageSeries_simple(ims,delay)
if (nargin < 2)
    delay = 0;
end


for k = 1:length(ims)
    k
    I = ims{k};
    clf; imagesc2(I); axis image;
    if (delay == 0)
        pause
    elseif (delay > 0)
        pause(delay);
    end
    drawnow
end
