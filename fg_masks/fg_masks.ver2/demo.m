%% configurtation - paths, parameters, etc
init;
config;

%% calculate global GIST/BOW distances between images.
[trainSet,testSet] = getImageSets(conf);
[IB,IG,dict] = getGlobalDistances(conf,trainSet,testSet);
%% get unary probability maps using both shape and appearance information
constructAllUnaryPotentials(conf,trainSet,testSet,IB,IG,dict);

%% construct binary potential values...
graphDatas = constructAllGraphData(conf,testSet);

%% apply graph cut on each image to get segmentation.
res = applyAllGraphCuts(conf,testSet,graphDatas,1);
% calculate best covering score for resulting segmentation
[bcs_u]=bestCoveringScore(conf.VOCopts,testSet,res);