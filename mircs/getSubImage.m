
function [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = ...
    getSubImage(conf,imgData,inflationFactor,aroundMouth,manualFace,needImage)
M = [];face_box = [];face_poly = [];mouth_box = [];mouth_poly = [];xy_c = [];
if (nargin < 3)
    inflationFactor = 1;
end
if (nargin < 4)
    aroundMouth = 1;
end
if (nargin < 5)
    manualFace = false;
end
if (nargin < 6)
    needImage = true;
end
conf.get_full_image = true;
% curIndex = findImageIndex(newImageData,currentID);
M = [];
% landmarks = newImageData(curIndex).faceLandmarks;
currentID = imgData.imageID;
landmarks = imgData.faceLandmarks;
if (imgData.faceScore == -1000)
    M = [];landmarks = [];
    % as a last resort, return the upper half of the bounding box.
    
    [I,I_rect] = getImage(conf,currentID);
    hh = I_rect(4)-I_rect(2);
    face_box = [I_rect([1 2 3]) I_rect(2)+hh/2];
    
    return;
end
%
[xy_c,mouth_poly,face_poly] = getShiftedLandmarks(landmarks,-imgData.I_rect);
mouth_box = pts2Box(mouth_poly);
face_box = pts2Box(face_poly);
face_sz = face_box(4)-face_box(2);
if (~aroundMouth),
    mouth_box = face_box;% use zhu/ramanan by default.
    if (imgData.faceScore <-.6)
        if (manualFace) % if training, use manually provided face.
            if (imgData.isTrain)
                mouth_box = makeSquare(imgData.gt_face.bbox);
            else
                mouth_box = imgData.alternative_face;
            end
            face_sz = mouth_box(4)-mouth_box(2);
        else
        end
    end
end
face_box = inflatebbox(mouth_box,face_sz*inflationFactor, 'both', true);
% clf;imagesc(I); axis image; hold on; plotPolygons(face_poly,'g','LineWidth',2);
% pause;
if (needImage)
    [I,I_rect] = getImage(conf,currentID);
    M = cropper(I,round(face_box));
end