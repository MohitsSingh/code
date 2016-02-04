function M = getRegionFeatureMatrix(imageDataStruct)
M = {};
for k = 1:length(imageDataStruct.imageIDs)
    imageInd = k
    
    currentID = imageDataStruct.imageIDs{imageInd};
    faceBoxShifted = imageDataStruct.faceBoxes(imageInd,:);
    lipRectShifted = imageDataStruct.lipBoxes(imageInd,:);
    lipTestBox = inflatebbox(lipRectShifted,[1 2],'post');
    %[regions,regionOvp,G] = getRegions(conf,currentID,false);
    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    ovp = boxesOverlap(lipTestBox,segmentBoxes)';
    sel_ = ovp > 0 & segmentBoxes(:,2) >= lipTestBox(2);
    %     imgArea = prod(dsize(I,1:2));
    %     areas = cat(1,props.Area);
    minorAxis = cat(1,props.MinorAxisLength);
    majorAxis = cat(1,props.MajorAxisLength);
    aspects = majorAxis./minorAxis;
    wh = faceBoxShifted(3:4)-faceBoxShifted(1:2); % width, height of the face box.
    minorAxisRatio = minorAxis/wh(1);
    orientations = cat(1,props.Orientation);
    sel_ = sel_ & aspects > 3 & minorAxisRatio < .1;
    
    r = [minorAxis,majorAxis,aspects,repmat(wh(1),size(majorAxis)),minorAxisRatio,orientations,sel_,k*ones(size(minorAxis))];
    
    M{k} = r;
end
M = cat(2,M);
