function res = fra_features_lite(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    %load s40_fra;
    %load ~/code/mircs/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
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


%[res.feats,res.moreData,res.selected_regions] = extract_all_features(reqInfo.conf,curImageData,params);
try
    [res.moreData] = extract_all_features_lite(reqInfo.conf,curImageData,params);
catch e
    warning('!!!')
    res.feats = [];
    res.moreData = [];
end
end