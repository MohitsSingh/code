function rects = getUpperBodyDets(conf,curImageData)
resPath = j2m(conf.upperBodyDir,curImageData.imageID);
load(resPath);
if (~isempty(res)) %#ok<*NODEF>
    res = res(:,[1:4 6]);
else
    res = [];
end
rects = res;
if (isempty(rects))
    return;
end
[I,I_rect] = getImage(conf,curImageData.imageID,true,true);

ints = BoxIntersection(rects,I_rect);
[~,~,areas] = BoxSize(rects);
[~,~,ints] = BoxSize(ints);
rects = rects(ints./areas >= .7,:);
rects = rects(nms(rects,.5),:);
end