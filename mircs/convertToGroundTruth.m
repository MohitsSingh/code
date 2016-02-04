function gt = convertToGroundTruth(pts,name)
gt = struct('name',{},'objectID',{},'occlusion',{},'representativeness',{},...
    'uncertainty',{},'deleted',{},'verified',{},'date',{},'sourceAnnotation',{},'polygon',{},...
    'objectParts',{},'comment',{},'sourceImage',{},'Orientation',{},'partID',{}');

t = 0;
for k = 1:length(pts)        
    curPts = pts(k).xy;
    for kk = 1:size(curPts,2)
        t = t+1;
        gt(t).name = name;
        gt(t).Orientation = 0;
        gt(t).sourceImage = pts(k).imgName;
        gt(t).polygon.x = curPts(1,kk);
        gt(t).polygon.y = curPts(2,kk);
    end        
end