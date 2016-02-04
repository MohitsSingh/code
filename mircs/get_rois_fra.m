function [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams)


if (nargin < 3)
    roiParams.infScale = 3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    roiParams.useCenterSquare = false;
    roiParams.squareSide = 30*roiParams.absScale/105;
end
if (~isfield(roiParams,'absScale'))
    roiParams.absScale = -1;
end
if (~isfield(roiParams,'useCenterSquare'))
    roiParams.useCenterSquare = false;
else
    if (roiParams.useCenterSquare && ~isfield(roiParams,'squareSide'))
        roiParams.squareSide = 30*roiParams.absScale/105;
    end
end

if (~isfield(roiParams,'centerOnMouth'))
    roiParams.centerOnMouth = true;
end


infScale = roiParams.infScale;
absScale = roiParams.absScale;
useCenterSquare = roiParams.useCenterSquare;
% if (nargin < 3)
%     infScale = 2.5;
% end
% if (nargin < 4)
%     absScale = 200;
% end


rois = struct('name',{},'id',{},'bbox',{});
if ~isfield(roiParams,'roiBox')
    faceBox = curImageData.faceBox;
    faceBox = makeSquare(faceBox);
else
    faceBox = roiParams.roiBox;
end
    
roiBox = round(inflatebbox(faceBox,infScale,'both',false));

if (roiParams.centerOnMouth)
    mouthCenter = curImageData.mouth(1,:);% if there's more than one mouth center...
    roiBox = round(roiBox + repmat(mouthCenter-boxCenters(roiBox),1,2));
end

[I,I_rect] = getImage(conf,curImageData);

I = cropper(I,roiBox);
if (absScale~=-1)
    scaleFactor = absScale/size(I,1);
else
    scaleFactor = 1;
end
I = imResample(I,scaleFactor);
n = 0;
FACE_ID = 1;
HAND_ID = 2;
OBJ_ID = 3;
MOUTH_ID = 4;
GLOBAL_ID = 5;

roi_offset = roiBox([1 2 1 2]);
n = n+1;
rois(n).name = 'face';
rois(n).id = FACE_ID;
rois(n).bbox = (curImageData.faceBox-roi_offset)*scaleFactor;

if (isfield(curImageData,'hands'))
    handBox = curImageData.hands;
    for tt = 1:size(handBox,1)
        %     if (~isempty(handBox)) % maybe no hands participate in action
        curHandBox = handBox(tt,:);
        %handBox = [min(handBox(:,1:2),[],1) max(handBox(:,3:4),[],1)]; % "union" of all boxes
        n = n+1;
        rois(n).name = 'hand';
        rois(n).id = HAND_ID;
        rois(n).bbox = (curHandBox-roi_offset)*scaleFactor;
        %     end
    end
end
if (isfield(curImageData,'objects'))
    objRoi = false(size2(I));
    if (~isempty(curImageData.objects))
        for ii = 1:length(curImageData.objects) %this is a bug, should use only one
            if ~(curImageData.objects(ii).toKeep)
                warning('get_rois_fra: skipping non-person related action object');
                continue
            end
            n = n+1;
            rois(n).name = 'obj';
            rois(n).id = OBJ_ID;
            rois(n).bbox = (pts2Box(curImageData.objects(ii).poly)-roi_offset)*scaleFactor;
            rois(n).poly = bsxfun(@minus,curImageData.objects(ii).poly,roi_offset(1:2))*scaleFactor;
        end
    end
end

if (isfield(curImageData,'mouth') && ~isempty(curImageData.mouth))
    for t = 1:size(curImageData,1)
        n = n+1;
        rois(n).name = 'mouth';
        bbox = repmat(curImageData.mouth(t,:),1,2);
        bbox = (bbox-roi_offset)*scaleFactor;
        bbox = inflatebbox(bbox,[40 40],'both',true); % since it's originally a point.
        rois(n).bbox = bbox;
        rois(n).id=MOUTH_ID;
    end
end

% for each of the regions (except the global one), modify to square if
% specified
if (roiParams.useCenterSquare)
    for t = 1:length(rois)
        rois(t).bbox = inflatebbox(rois(t).bbox,roiParams.squareSide,'both',true);
    end
end

% add the entire area as a roi.
n = n+1;
rois(n).name = 'global';
rois(n).bbox = [1 1 size(I,2) size(I,1)];
rois(n).id=GLOBAL_ID;