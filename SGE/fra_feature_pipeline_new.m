function res = fra_feature_pipeline_new(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    addpath(genpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained'));install
    addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
    addpath(genpath('/home/amirro/code/3rdparty/attribute_code'));
    config;    
    res.conf = conf;
    params = defaultPipelineParams();                
    params.testMode = true;
    params.keepSegments = false;    
    res.params = params;
    return;
end


params = reqInfo.params;
images_db = params.images_db;
imgIndex = findImageIndex(images_db,I);
curImageData = images_db(curImageData);
[I_orig,I_rect] = getImage(reqInfo.conf,curImageData);
%[I_orig,I_rect] = getImage(reqInfo.conf,I);
curImageData = face_detection_to_fra_struct(conf,moreParams.pipelineParams(1).outDir,I,moreParams.pipelineParams(2).outDir);
curImageData.faceBox = curImageData.faceBox+I_rect([1 2 1 2]);

% if (moreParams.forceTestMode)
%     params.testMode = true;
%     params.keepSegments = true;
% end

if (isfield(moreParams,'params'))
    params = moreParams.params;
end

params.pipelineParams = moreParams.pipelineParams;
params.externalDB = true;
[res.feats,res.moreData,res.selected_regions] = extract_all_features(reqInfo.conf,curImageData,params);
res.curImageData = curImageData;
end