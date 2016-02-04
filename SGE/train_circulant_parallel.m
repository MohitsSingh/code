function train_circulant_parallel(clusters,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
conf = extraInfo.conf;
naturalSet = extraInfo.naturalSet;
path(extraInfo.path);
%TODO: replace this with your own implementation code
% conf,discovery_set,clusters,iter,suffix,false
res = train_circulant(conf,clusters,naturalSet);
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');