function [res] = extract_all_features_2(imgData,params,initData)
%Aggregates multiple feature types for images. Some features may have been
% precomputed

res = struct('type',{},'feat',{});
cacheDir = params.cacheDir;
globalFeaturesDir = fullfile(cacheDir,'global_feats');
boxFeaturesDir = fullfile(cacheDir,'local_feats');
faceDetDir = fullfile(cacheDir,'face_det');
faceFeaturesDir = fullfile(cacheDir,'face_feats');
% get global features and features from person bounding box
ensuredir(globalFeaturesDir);
ensuredir(boxFeaturesDir);
ensuredir(faceDetDir);
ensuredir(faceFeaturesDir);


I = imread2(imgData.image_path);

globalFeaturesPath = j2m(globalFeaturesDir,imgData.image_id);
n = 0;
if (~exist(globalFeaturesPath,'file'))
    global_feats = getGlobalFeatures(I,initData,params);
    save(globalFeaturesPath,'global_feats');
else load(globalFeaturesPath); end
n = n+1;
res(n).type = 'global';
res(n).feat = global_feats;
bbox = imgData.objects.bbox;
I_crop = cropper(I,bbox);

localFeaturesPath = j2m(boxFeaturesDir,[imgData.image_id '_' num2str(imgData.idx)]);
if (~exist(localFeaturesPath,'file'))
    local_features = getGlobalFeatures(I_crop,initData,params);
    save(localFeaturesPath,'local_features');
else load(localFeaturesPath); end

n = n+1;
res(n).type = 'box';
res(n).feat = local_features;

% detect faces.
facesPath = j2m(faceDetDir,[imgData.image_id '_' num2str(imgData.idx)]);
if (~exist(facesPath,'file'))
    faceInfo = getFaceData(initData, I, bbox);
    save(facesPath,'faceInfo');
else load(facesPath); end

%faceInfo = getFaceInfo(I, bbox, j2m(faceDetDir,[imgData.image_id '_' num2str(imgData.idx)]));

faceFeaturesPath = j2m(faceFeaturesDir,[imgData.image_id '_' num2str(imgData.idx)]);
if (~exist(faceFeaturesPath,'file'))
    faceFeatures = struct;
    [ faceFeatures ] = featuresFromHead(I, faceInfo, initData.net);
    %     f = 0;
    %     faceRegions = faceInfo.faceRegions;
    %     for iRegion = 1:length(faceRegions)
    %         f = f+1;
    %         curBox = faceRegions(iRegion).bbox;
    %         faceFeatures(f).type = 'box';
    %         faceFeatures(f).feat = getGlobalFeatures(cropper(I,bbox),initData);
    %     end
    save(faceFeaturesPath,'faceFeatures');
else load(faceFeaturesPath); end

res = [res,faceFeatures];

function faceInfo = getFaceData(initData,I_orig,I_rect)
% detect a face + landmarks
model = initData.model;
resizeFactor = 2;
I = cropper(I_orig,I_rect);
I = imresize(I,resizeFactor,'bilinear');
[ds, bs] = imgdetect(I, model,-1);
top = nms(ds, 0.1);
if (isempty(top))
    boxes = -inf(1,5);
end
boxes = ds(top(1:min(1,length(top))),:);
if (~isempty(boxes))
    boxes(:,1:4) = boxes(:,1:4)/resizeFactor;
else
    boxes = [I_rect(1:3) (I_rect(2)+I_rect(4))/2 -10];
end

boxes(:,1:4) = bsxfun(@plus,boxes(:,1:4),I_rect([1 2 1 2])-1);

landmarkInit = initData.landmarkParams;
conf.get_full_image = false;
I = cropper(I_orig,round(boxes));
resizeFactor = 128/size(I,1);
I = imResample(I,[128 128],'bilinear');
bb = [1 1 fliplr(size2(I))];
landmarkInit.debug_ = false;
[kp_global] = myFindFacialKeyPoints(conf,I,bb,landmarkInit.XX,...
    landmarkInit.kdtree,landmarkInit.curImgs,landmarkInit.ress,landmarkInit.ptsData,landmarkInit);
kp_global(:,1:4) = kp_global(:,1:4)/resizeFactor;
kp_global(:,1:4) = bsxfun(@plus,kp_global(:,1:4),boxes([1 2 1 2])-1);

faceInfo.faceBox = boxes;
faceInfo.landmarks = [boxCenters(kp_global(:,1:4)) kp_global(:,end)];

function r = getGlobalFeatures(I,initData,params)
r= extractDNNFeats(I,initData.net,params.layers);
r = r.x;