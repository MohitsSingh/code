function res = action_feature_extraction(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    % inialize parameters for various modules:
    % facial landmark parameters
    landmarkParams = load('~/storage/misc/kp_pred_data.mat');
    landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX);
    landmarkParams.conf = conf;
    landmarkParams.wSize = 96;
    landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
    landmarkParams.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    
    % segmentation...
    addpath '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;
    nn_net = init_nn_network();
    clear predData;
    
    %% object prediction data
    %     load ~/code/mircs/s40_fra.mat;
    objPredData = load('~/storage/misc/actionObjectPredData.mat');
    objPredData.kdtree = vl_kdtreebuild(objPredData.XX,'Distance','L2');
    params = defaultPipelineParams(false);
    
    %%
    
    dataDir = '~/storage/s40_fra_feature_pipeline_stage_1';
    params.dataDir = dataDir;
    curParams = params;  
    clear all_results;       
    curParams.landmarkParams = landmarkParams;
    curParams.features.nn_net = nn_net;
    curParams.objPredData = objPredData;
    
    res.featureParams = curParams;
    res.conf = conf;
    
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    res.fra_db = s40_fra_faces_d;
    return;
end

conf = initData.conf;
featureParams = initData.featureParams;
fra_db = initData.fra_db;
k = findImageIndex(fra_db,params.name);
[res.regionFeats,res.imageFeats] = extract_all_features(conf,fra_db(k),featureParams);
