function [ fg_mask] = pasRec2Mask( VOCopts,currentID,rec,outputSize)
%PASREC2MASK Summary of this function goes here
%   Detailed explanation goes here
if (nargin < 3) % record already given
    rec = PASreadrecord(sprintf(VOCopts.annopath,currentID));
end
sizeRatio = [1 1];
if (nargin == 4)
    fg_mask = false(outputSize);
    sizeRatio = outputSize./[rec.size.height,rec.size.width];
else
    fg_mask = false(rec.size.height,rec.size.width);
end
for iObj = 1:length(rec.objects)
    curBox = rec.objects(iObj).bndbox;
    ymin = max(1,round(curBox.ymin*sizeRatio(1)));
    ymax = floor(curBox.ymax*sizeRatio(1));
    xmin = max(1,round(curBox.xmin*sizeRatio(2)));
    xmax = floor(curBox.xmax*sizeRatio(2));
    fg_mask(ymin:ymax,xmin:xmax) = true;
end

end

