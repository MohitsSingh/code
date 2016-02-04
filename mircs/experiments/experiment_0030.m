%% %%%%% Experiment 0030 %%%%%
% 13/2/2014 : try to apply piotr dollar's code : 
% conclusions so far - not as good as exepcted, probably trained for non-side views. 

% initialization of paths, configurations, etc

if (~exist('initialized','var'))
    initpath;
    config;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/misc/imageData_new;
end

curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
extraInfo.models = models;
extraInfo.path = path;
%extraInfo.dpmVersion = 4;
extraInfo.newImageData = newImageData;
% delete ~/sge_parallel_new/*;
job_suffix = 'cpmc';
justTesting = false;
outDir = '~/storage/s40_cpmc';
detections = run_and_collect_results({newImageData.imageID},'cpmc_parallel',justTesting,extraInfo,job_suffix,[],outDir);
save(fullfile(outDir,'all.mat'),'detections');