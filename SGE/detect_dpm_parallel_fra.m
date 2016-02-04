function res = detect_dpm_parallel_fra(conf,I,reqInfo)

if (nargin == 0)
    load ~/code/mircs/dpm_models/fra_models.mat;
    res.models = all_models;
    addpath('~/code/mircs');
    cd ~/code/mircs/;
    initpath;
    config;
    res.conf = conf;
    cd ~/code/3rdparty/voc-release5/;
    startup;
    return;
end
models = reqInfo.models;
%
load ~/code/mircs/fra_db.mat; % silly but will work
[~,name,ext] = fileparts(I);
imgData = fra_db(findImageIndex(fra_db,[name ext]));
[rois,roiBox,I] = get_rois_fra(conf,imgData);

% I = imread(I);
% faceBox = inflatebbox(imgData.faceBox,2.5,'both',false);
% I = cropper(I,round(faceBox));
% resizeFactor = 256/size(I,1);
% I = imResample(I,resizeFactor,'bilinear');
res = struct('class',{},'boxes',{});
cd ~/code/3rdparty/voc-release5/;
for iModel = 1:length(models)
    res(iModel).class = models(iModel).class;
    [responses, bs] = imgdetect(I, models(iModel),-1.1);
    if (~isempty(responses))
        responses = responses(:,[1:4 6]);
        responses = responses(nms(responses,.8),:); % mild non-maximal suppression.
        %         responses(:,1:4) = responses(:,1:4)/resizeFactor;
        res(iModel).boxes =responses;
    end
end

detections = res;
res = struct('detections',detections);