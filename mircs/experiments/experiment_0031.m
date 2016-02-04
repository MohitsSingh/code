%% %%%%% Experiment 0031 %%%%%
% 13/3/2014 :
% get CPMC on all images...
if (~exist('initialized','var'))
    initpath;
    config;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/misc/imageData_new;
    newImageData = augmentImageData(conf,newImageData);
    addpath('~/code/3rdparty/cpmc_release1/');
    addpath(pwd);
    addpath(fullfile(pwd,'utils'));
end

curDir = pwd;
extraInfo.path = path;
cd ~/code/SGE
extraInfo.conf = conf;

%extraInfo.dpmVersion = 4;
extraInfo.newImageData = newImageData;

extractInfo.minFaceScore = -.6;
% extraInfo.runMode = 'upperBody';
extraInfo.runMode = 'sub';
% delete ~/sge_parallel_new/*;
job_suffix = 'cpmc_u';
justTesting = true;

extraInfo.newImageData = newImageData(~[newImageData.isTrain]);

outDir = '~/storage/s40_cpmc_u';
results = run_and_collect_results({extraInfo.newImageData.imageID},'cpmc_parallel',justTesting,extraInfo,job_suffix,50,outDir);
save(fullfile(outDir,'all.mat'),'results');

%% gpb on face areas.
curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
extraInfo.path = path;
%extraInfo.dpmVersion = 4;
extraInfo.newImageData = newImageData;
extractInfo.minFaceScore = -.6;
extraInfo.runMode = 'sub';
% delete ~/sge_parallel_new/*;
job_suffix = 'gpb_faces';
justTesting = false;

extraInfo.scale = 1;
extraInfo.absScale = [];


extraInfo.newImageData = newImageData([newImageData.label]);

extractInfo.minFaceScore = -1;
outDir = '~/storage/gpb_faces_new';
results = run_and_collect_results({extraInfo.newImageData.imageID},'gpb_parallel_new',justTesting,extraInfo,job_suffix,50,outDir);
save(fullfile(outDir,'all.mat'),'results');


for k = 1:length(newImageData)
    k = 1164
    [regions,~,G] = getRegions(conf,newImageData(k).imageID,false,outDir);
    [M,~,face_box,face_poly] = getSubImage(conf,newImageData(k),2,false);
    displayRegions(M,regions);
    [ucm,gPb_thin] = loadUCM(conf,curImageData.imageID,outDir); %#ok<*STOUT>
    imshow(ucm)
end


