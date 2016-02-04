function [faceLandmarks,allBoxes_complete,faceBoxes_complete] = landmarks2struct(landmarks,images,dets,conf)

% faceLandmarks = struct('s',{},'c',{},'xy',{},'level',{},'lipBox',{},'faceBox',{},'dpmRect',{},'dpmModel',{});
% faceLandmarks = orderfields(repmat(faceLandmarks,length(landmarks),1));
faceLandmarks = [];
for q = 1:length(landmarks)
    curLandmark = landmarks(q);  
    curLandmark.lipBox = [];
    curLandmark.faceBox = [];
       
    ptCenters = polyCenters(curLandmark.polys);
    nPoints = size(ptCenters,1);
    if (nPoints==68)
        lipCoords = ptCenters(33:51,:);
        lipBox = pts2Box(lipCoords);
    elseif (nPoints == 39)
        lipCoords = ptCenters(16:22,:);
        lipBox = pts2Box(lipCoords);
    else
        error('unexpected number of facial keypoints');        
    end
    faceBox = pts2Box(boxCenters(curLandmark.xy));
    faceLandmarks(q) = orderfields(curLandmark);
    faceLandmarks(q).lipBox = lipBox;
    faceLandmarks(q).faceBox = faceBox;
    
end

function pc = polyCenters(polys)
pc = cellfun2(@(x) mean(x,1),polys);
pc = cat(1,pc{:});


