function [landmarks] = landmarks2struct_3(landmarks,offset)
if (nargin < 2)
    offset = [0 0];
end
for q = 1:length(landmarks)
    curLandmark = landmarks(q);
    curLandmark.lipBox = [];
    curLandmark.faceBox = [];    
    
    polys = cellfun2(@(x) bsxfun(@plus,x,offset),curLandmark.polys);
    ptCenters = polyCenters(curLandmark.polys);
%     ptCenters = bsxfun(@plus,ptCenters,offset);
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
    faceBox = pts2Box(ptCenters);
    landmarks(q).lipBox = lipBox;
    landmarks(q).faceBox = faceBox;
    
end

function pc = polyCenters(polys)
pc = cellfun2(@(x) mean(x,1),polys);
pc = cat(1,pc{:});

