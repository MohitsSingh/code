function classify_parallel(baseDir,d,indRange,outDir)

cd /home/amirro/code/mircs;

initpath;
config;
[train_ids,train_labels] = getImageSet(conf,'train');
[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);

initUnaryModels;
% faceModel = learnFaceModel_new(conf);

% models = [partModels,faceModel]
models = partModels;
% conf.get_full_image = true;
% [learnParams,conf] = getDefaultLearningParams(conf);
% learnParams.partNames = partNames;
% partModels = learnModels2(conf,train_ids,train_labels,groundTruth,learnParams);

for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    %     fprintf('checking if results for image %s exist...',filename);
    if (exist(resFileName,'file'))
        %         fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating...');
    end
    
    regionConfs = struct('score',{},'pred',{});
        
    for iModel = 1:length(models)
        m = models(iModel);
        feats = m.extractor.extractFeatures(currentID);
        %     w = m.models.w;
        [regionConfs(iModel).pred, regionConfs(iModel).score] = m.models.test(feats);
        %     save(resPath,'regionConfs');
    end
    
    %     [regionConfs] = applyModel(conf,currentID,partModels);
    save(resFileName,'regionConfs');
    %     fprintf('\tdone!\n');
end
% fprintf('\n\n\nFINISHED\n\n\n!\n');
end