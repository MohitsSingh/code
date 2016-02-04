 function target = shiftImage(subImage,bbox,target)
    [rows cols] = BoxSize(bbox([2 1 4 3]));
    subImage = imResample(subImage,[rows cols],'bilinear');
    target(bbox(2):bbox(4),bbox(1):bbox(3),:) = subImage;
 end