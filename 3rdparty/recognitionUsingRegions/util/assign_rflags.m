function regionflags = assign_rflags(rects,regions,frac)
% function regionflags = assign_rflags(rects,regions,frac)
%
% Copyright @ Chunhui Gu, April 2009

rng.x = zeros(length(regions),2);
rng.y = zeros(length(regions),2);
for rId = 1:length(regions),
    [y,x] = find(regions{rId}==1);
    rng.x(rId,:) = [min(x) max(x)];
    rng.y(rId,:) = [min(y) max(y)];
end;

nrects = size(rects,1);
imsz = [size(regions{1},1) size(regions{1},2)];
regionflags = false(nrects,length(regions));
for rectId = 1:nrects,
    mask = getmask(rects(rectId,:),imsz);
    for rId = 1:length(regions),
        if rects(rectId,1)<rng.x(rId,2) && rects(rectId,1)+rects(rectId,3)>rng.x(rId,1) ...
                && rects(rectId,2)<rng.y(rId,2) && rects(rectId,2)+rects(rectId,4)>rng.y(rId,1),
            int = mask & regions{rId};
            if sum(int(:)) / sum(regions{rId}(:)) > frac,
                regionflags(rectId,rId) = true;
            end;
        end;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mask = getmask(rect,imsz)

mask = false(imsz);
rect = round(rect);
mask(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = true;