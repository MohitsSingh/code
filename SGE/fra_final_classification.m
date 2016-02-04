function res = fra_final_classification(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    %load s40_fra;
    load ~/code/mircs/s40_fra_faces_d.mat;
    res.fra_db = s40_fra_faces_d;
    res.conf = conf;
    %     params = defaultPipelineParams();
    %     params.features.dnn_net = init_nn_network(true);
    load ~/code/mircs/classifier_all.mat;
    res.classifier_all = classifier_all;
    
    return;
end

fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
params = moreParams;


res.classificationResult = applyToImageSet(fra_db(k),reqInfo.classifier_all.classifiers,params);
resPath = j2m(params.dataDir,fra_db(k));
load(resPath,'moreData');
res.moreData = moreData;
%[res.feats,res.moreData,res.selected_regions] = extract_all_features(reqInfo.conf,curImageData,params);

end