function [faceLandmarks,allBoxes_complete,faceBoxes_complete] = landmarks2struct(landmarks,images,dets,conf)

faceLandmarks = struct('s',{},'c',{},'xy',{},'level',{},'lipBox',{},'faceBox',{})
faceLandmarks(1).s = [];
faceLandmarks = repmat(faceLandmarks,length(landmarks),1);
for q = 1:length(landmarks)
    bs = landmarks{q};
    if (isempty(bs))
        faceLandmarks(1).s = -1000;
        continue;
    end
    
%     bs(1).xy = bs(1).xy/2;
    bs(1).lipBox = [];
    bs(1).faceBox = [];
    
%     if size(bs(1).xy,1) < 64
%         continue;
%     end

%     if (dets(1).cluster_locs(q,conf.consts.FLIP))
%         [rows cols ~] = BoxSize(dets(1).cluster_locs(q,:));
%          bs(1).xy = flip_box(bs(1).xy,[rows cols])
%     end
% 
    if (size(bs(1).xy,1) >= 51)
        lipCoords = boxCenters(bs(1).xy(33:51,:));
        lipBox = pts2Box(lipCoords);
    elseif (size(bs(1).xy,1) == 39)
        lipCoords = boxCenters(bs(1).xy([20:22 28 29],:));
        lipBox = pts2Box(lipCoords);
    else
        lipCoords = [];
        lipBox = [];
    end
    faceBox = pts2Box(boxCenters(bs(1).xy));
    
    faceLandmarks(q) = bs(1);
    faceLandmarks(q).lipBox = lipBox;
    faceLandmarks(q).faceBox = faceBox;
        
end


% for images where the lips were not detected,
% take the average box.
allBoxes_missing = cat(1,faceLandmarks.lipBox);
faceBoxes_missing = cat(1,faceLandmarks.faceBox);
meanBox = mean(allBoxes_missing,1);
meanFaceBox = mean(faceBoxes_missing,1);
minScore = min([faceLandmarks.s]);
for k = 1:length(faceLandmarks)
    if (isempty(faceLandmarks(k).lipBox))
        k
        faceLandmarks(k).lipBox = meanBox;
        faceLandmarks(k).faceBox = meanFaceBox;
        faceLandmarks(k).c = 0;
    end
    
    if (isempty(faceLandmarks(k).s))
        faceLandmarks(k).s = minScore;
        
    end
end

allBoxes_complete = cat(1,faceLandmarks.lipBox);
faceBoxes_complete = cat(1,faceLandmarks.faceBox);