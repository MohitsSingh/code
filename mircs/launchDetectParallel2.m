%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function detections = launchDetectParallel2(conf,clusters,imagePaths,job_suffix)
%% Part 1: General Usage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
extraInfo.clusters = clusters;
extraInfo.path = path;
% delete ~/sge_parallel_new/*;
detections = run_and_collect_results(imagePaths,'detect_parallel',false,extraInfo,job_suffix);
% detections = cat(1,detections{:});
% imageIndices = cat(1,imageIndices{:});
% [s,is] = sort(imageIndices);
% detections = detections(is);
cd(curDir);

