[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
[learnParams,conf] = getDefaultLearningParams(conf,256);
learnParams.useS40Negatives = true;

if (~learnParams.useS40Negatives)
    nonPersonIds = getNonPersonIds(conf.VOCopts);
    for k = 1:length(nonPersonIds)
        nonPersonIds{k} = [nonPersonIds{k} '.jpg'];
    end
    learnParams.negImages = nonPersonIds;
end

learnParams.exclusiveLabels = false;
learnParams.partNames = partNames;
learnParams.doHardNegativeMining = true;
learnParams.useRealGTSegments = false;
learnParams.nNegativeMiningRounds = 20;
learnParams.featureExtractors{1}.useRectangularWindows = false;
learnParams.debugSuffix = '256_neg_context_hn';
learnParams.balanceDatasets = true;

imageData = initImageData;
learnParams.partNames = {'cup'};
partModels = learnModels3(conf,groundTruth,learnParams,imageData);



% special treatment for cup.
% partModels = learnModels3(conf,train_ids,train_labels,groundTruth,learnParams);
