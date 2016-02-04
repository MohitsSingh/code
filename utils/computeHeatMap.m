function [map,counts] = computeHeatMap(img,windows,mode)
if (nargin < 3)
    mode = 'sum';
end
sz = size2(img);
if (strcmp(mode,'max'))
    map = -inf(sz);
else
    map = zeros(sz);
end
counts = zeros(sz);
if (size(windows,2)<5)
    windows(:,5) = 1;
end
for idx = 1:size(windows,1)
    %     idx
    xmin = uint16(round(windows(idx,1)));
    ymin = uint16(round(windows(idx,2)));
    xmax = uint16(round(windows(idx,3)));
    ymax = uint16(round(windows(idx,4)));
    
    bb = clip_to_image([xmin ymin xmax ymax],[1 1 fliplr(sz)]);
    xmin = bb(1); ymin = bb(2); xmax = bb(3); ymax = bb(4);
    score = windows(idx,5);
    if (strcmp(mode,'max'))
        maskBox = -inf(sz);
    else
        maskBox = zeros(sz);
    end
    maskBox(ymin:ymax,xmin:xmax) = score;
    counts(ymin:ymax,xmin:xmax) = counts(ymin:ymax,xmin:xmax)+1;
    if (strcmp(mode,'max'))
        map = max(map,maskBox);
    else
        map = map + maskBox;
    end
    
end
if (strcmp(mode,'max'))
    infs = isinf(map);
    map(infs) = min(map(~infs))-std(map(~infs));
end
end