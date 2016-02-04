function region_features_parallel(indRange,outDir,isTrain)
if (isTrain)
    resFileName = fullfile(outDir,sprintf('%06.0f_train.mat',indRange(1)));
else
    resFileName = fullfile(outDir,sprintf('%06.0f_test.mat',indRange(1)));
end

if (exist(resFileName,'file'))
    disp(['result file ' resFileName ' already exists']);
    return;
end
cd ~/code/mircs;
initpath;
config;
addpath('/home/amirro/code/3rdparty/sliding_segments');
load ~/storage/misc/imageData_new; % which image-data to load? the one by zhu, or my face detection + mouth detection?

if (isTrain)
    load ~/mircs/experiments/experiment_0015/regionData_train.mat
    regionData = regionData_train;
    imageSet = imageData.train;
    
else
    load ~/mircs/experiments/experiment_0015/regionData_test.mat
    regionData = regionData_test;
    imageSet = imageData.test;
end
% TODO - remember that you set the min face score to -.7 here. 
feats = extractRegionFeatures(conf,regionData,imageSet,false,indRange,-.7);

save(resFileName,'feats');
end