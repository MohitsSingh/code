function detect_in_roi_parallel(imageIDs,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
clusters = extraInfo.clusters;
conf = extraInfo.conf;
newImageData = extraInfo.newImageData;
path(extraInfo.path);
iter = 0;

res = {};
for k = 1:length(imageIDs)
    k   
    currentID = imageIDs{k};    
    res{k} = detect_in_roi(conf,clusters,newImageData,currentID);
end
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');