function res = feat_pipeline(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    addpath('~/code/SGE');
    cd ~/code/mircs;
    addpath('~/code/mircs');
    initpath;
    config;
    conf.get_full_image = true;
    roiParams = defaultROIParams();
    landmarkParams = load('~/storage/misc/kp_pred_data.mat');
    ptNames = landmarkParams.ptsData;
    ptNames = {ptNames.pointNames};
    requiredKeypoints = unique(cat(1,ptNames{:}));
    landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX,'Distance','L2');
    landmarkParams.conf = conf;
    landmarkParams.wSize = 96;
    landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
    landmarkParams.requiredKeypoints = requiredKeypoints;
    landmarkInit = landmarkParams;
    landmarkInit.debug_ = false;
    nn_net = init_nn_network();
    cd /home/amirro/code/3rdparty/voc-release5
    startup
    load ~/code/3rdparty/dpm_baseline.mat
    res = struct('landmarkParams',landmarkParams,'net',nn_net,'model',model);
    return;
end

p.cacheDir = '~/storage/voc_2012_data_deep';
p.useDeepFeats = true;
res=struct('success',false);

extract_all_features_2(params.imgData,p,initData);

res.success = true;

end