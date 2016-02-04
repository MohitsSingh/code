function partModelsESVM = ...
    learnModelsESVM(conf,train_ids,train_labels,groundTruth,partNames)
partModelsESVM = struct('models',{},'M',{});
negImages = {};
for k = 1:length(train_ids)
    %         k
    if (~train_labels(k))
        negImages{end+1} = getImagePath(conf,train_ids{k});
    end
end
for k = 1:length(groundTruth)
    groundTruth(k).sourceImage = getImagePath(conf,groundTruth(k).sourceImage);
end

negImages = vl_colsubset(negImages,100);
mkdir('esvm_models');
for k = 1:length(partNames)
    modelPath = ['~/code/mircs/esvm_models/' conf.classes{conf.class_subset} '_' partNames{k} '.mat'];
    if (exist(modelPath,'file'))
        load(modelPath);
        
    else
        gt_data = groundTruth([groundTruth.partID] == k);
        curDir = pwd;
        cd /home/amirro/code/3rdparty/exemplarsvm/;
        [models,M] = esvm_demo_train_s40(gt_data,negImages)
        save(modelPath,'models','M');
    end
    partModelsESVM(k).models = models;
    partModelsESVM(k).M = M;
    
    
end
%     groundTruth.sourceImage
end