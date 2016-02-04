
function objInteractions = getObjectInteractions(groundTruth)
objInteractions = struct('sourceImage',{},'id1',{},'id2',{},'roi1','roi2');
gtInds = zeros(size(groundTruth));
gtInds(1) = 1;
% split ground-truth according to source images.
count_ = 1;
for k = 2:length(groundTruth)
    if (~strcmp(groundTruth(k).sourceImage,groundTruth(k-1).sourceImage)) % still the same.
        count_ = count_+1;
    end
    gtInds(k) = count_;
end
u = unique(gtInds);
count_ = 1;
for k = 1:length(u)
    currentInds = find(gtInds==u(k));
    curObjects = groundTruth(currentInds);
    for i1 = 1:length(curObjects)
        [x1,y1] = poly2cw(curObjects(i1).polygon.x,curObjects(i1).polygon.y);
        for i2 = setdiff(1:length(curObjects),i1)%length(curObjects)
            [x2,y2] = poly2cw(curObjects(i2).polygon.x,curObjects(i2).polygon.y);
            objInteractions(count_).sourceImage = groundTruth(currentInds(1)).sourceImage;
            objInteractions(count_).id1 = curObjects(i1).partID;
            objInteractions(count_).id2 = curObjects(i2).partID;
            objInteractions(count_).roi1 = [x1 y1];
            objInteractions(count_).roi2 = [x2 y2];
            count_=count_+1;
        end
    end
end


end

