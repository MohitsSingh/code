function displayImageSeries2(ims,delay)
if (nargin < 2)
    delay = 0;
end

for k = 1:length(ims)
%     k
    I = ims{k};
    if ischar(I)
        I = imread(I);
    end
    clf; imagesc2(I);
    if (delay == 0)
        pause
    elseif (delay > 0)
        pause(delay);
    end
    drawnow
end
end