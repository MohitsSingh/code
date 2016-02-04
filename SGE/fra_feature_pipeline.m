function res = fra_feature_pipeline(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    addpath(genpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained'));install
    addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
    addpath(genpath('/home/amirro/code/3rdparty/attribute_code'));
    config;
    %load s40_fra;
    load ~/code/mircs/s40_fra_faces_d.mat;
    res.fra_db = s40_fra_faces_d;
    res.conf = conf;
    params = defaultPipelineParams();
    params.features.dnn_net = init_nn_network(false);
    res.params = params;
    return;
end

fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
params = reqInfo.params;

if (isfield(moreParams,'params'))
    params = moreParams.params;
end

if (isfield(moreParams,'testMode'))
    params.testMode = moreParams.testMode;
end

%[res.feats,res.moreData,res.selected_regions] = extract_all_features(reqInfo.conf,curImageData,params);
try 
[res.feats,res.moreData] = extract_all_features(reqInfo.conf,curImageData,params);
catch e 
    warning('!!!')
    res.feats = [];
    res.moreData = [];
end

end