function [faceLandmarks,allBoxes_complete,faceBoxes_complete] = collectResults(conf,image_ids,landmarkDir)

conf.get_full_image = false;
landmarks = {};
if (nargin < 3)
    landmarkDir = '~/storage/landmarks_s40_new_big_2';
end
for k = 1:length(image_ids)
    k
    currentID = image_ids{k};
    L = load(fullfile(landmarkDir,strrep(currentID,'.jpg','.mat')));
    if (isempty(L.landmarks))
        continue;
    end
    % find the best scoring candidate.
    %     [curImage,xmin,xmax,ymin,ymax] = getImage(conf,image_ids{k});
    bb =[];
    for ii = 1:length(L.landmarks)
        curRes = L.landmarks(ii).results;
        for jj = 1:length(curRes)
            curRes(jj).dpmRect = L.landmarks(ii).dpmRect;
            curRes(jj).dpmModel = ii;
            curRes(jj).xy = fixBoxes(curRes(jj).xy,curRes(jj).dpmRect);
        end
        L.landmarks(ii).results = curRes;
        %bb = [bb;repmat(L.landmarks(ii).dpmRect,length(curRes),1)];
    end
        
    curLandmarks = [L.landmarks.results];
    if (~isempty(curLandmarks))
        dpmScores = cat(1,curLandmarks.dpmRect);
        dpmScores = dpmScores(:,6);
        [s,is] = sort([curLandmarks.s]+0*dpmScores','descend');
    else
        curLandmarks.s = -1000;
        curLandmarks.c = -1;
        curLandmarks.xy = [];
        curLandmarks.level= -1;
        curLandmarks.dpmRect = L.landmarks(1).dpmRect;
        curLandmarks.dpmModel = -1;
        is = 1;
    end
    %     curLandmarks(is(1)).xy = fixBoxes(curLandmarks(is(1)).xy,curLandmarks(is(1)).bb);
    
    landmarks{k} = curLandmarks(is(1));
% %     [curImage,xmin,xmax,ymin,ymax] = getImage(conf,image_ids{k});
% %          
% %             for ii = 1:length(is)
% %                 ii
% %                 clf; imagesc(curImage); axis image; hold on;
% %                 xy = landmarks{k}.xy;
% %                 plotBoxes2(xy(:,[2 1 4 3]),'g');
% %                 pause;
    
%             end
end
[faceLandmarks,allBoxes_complete,faceBoxes_complete] = landmarks2struct2(landmarks,image_ids,[],conf);
end
function rects = fixBoxes(rects,referenceRect)
rects = rects+repmat(referenceRect([1 2 1 2]),size(rects,1),1);
end
