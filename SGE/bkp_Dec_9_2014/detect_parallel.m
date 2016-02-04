function detect_parallel(imagePaths,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
clusters = extraInfo.clusters;
conf = extraInfo.conf;
path(extraInfo.path);
iter = 0;
%TODO: replace this with your own implementation code
% conf,discovery_set,clusters,iter,suffix,false
res = getDetections(conf,imagePaths,clusters,iter,0,false);
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');