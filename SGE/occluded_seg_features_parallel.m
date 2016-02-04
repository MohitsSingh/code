function classify_parallel(baseDir,d,indRange,outDir)

cd /home/amirro/code/mircs;

initpath;
config;
conf.get_full_image = true;
[learnParams,conf] = getDefaultLearningParams(conf,1024);

for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    
    if (exist(resFileName,'file'))
        %         fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating...');
    end
    [regions,region_sel] = getRegionSubset(conf,currentID,1,true);
    feats = learnParams.featureExtractors{1}.extractFeatures(currentID,regions);
    
    save(resFileName,'feats','regions','region_sel');
    fprintf('\tdone witn current image!\n');
end
fprintf('\n\n\nFINISHED ALL IMAGES IN CURRENT BATCH \n\n\n!\n');
end