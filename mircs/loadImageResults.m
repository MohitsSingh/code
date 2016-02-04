% % % function imageResults = loadImageResults(conf,imageID,newImageData)
% % %     fileName = strrep(imageID,'.jpg','.mat');
% % %     imageIndex = findImageIndex(newImageData,imageID);
% % %     % occluders
% % %     imageResults.occluders = load(fullfile(conf.occludersDir,fileName));
% % %     % hands
% % %     handRes = load(fullfile(conf.handsDir,fileName));
% % %     boxes = handRes.shape.boxes;
% % %     boxes = boxes(nms(boxes,.1),:);
% % %     imageResults.handLocations = boxes;
% % %     % regions
% % %     imageResults.regions = getRegions(conf,imageID);
% % %     imageResults.imageData = newImageData(imageIndex);
% % % %     imageResults.imageData = rmfield(imageResults.imageData,'occluders');
% % % end


function imageResults = loadImageResults(conf,imageID,newImageData)
fileName = strrep(imageID,'.jpg','.mat');
imageIndex = findImageIndex(newImageData,imageID);
% occluders

imageResults.regions = getRegions(conf,imageID,false);
[occlusionPatterns,dataMatrix,regions,face_mask] = ...
    getOcclusionPattern(conf,newImageData(imageIndex),imageResults.regions);
I = getImage(conf,imageID);
% imageResults.occluders = load(fullfile(conf.occludersDir,fileName));
% hands
handRes = load(fullfile(conf.handsDir,fileName));
boxes = handRes.shape.boxes;
boxes = boxes(nms(boxes,.1),:);
imageResults.handLocations = boxes;
% regions
imageResults.imageData = newImageData(imageIndex);
%     imageResults.imageData = rmfield(imageResults.imageData,'occluders');
end