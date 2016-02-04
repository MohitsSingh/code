[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
partNames = partNames(1:4);
[learnParams,conf] = getDefaultLearningParams(conf,1024);
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
learnParams.nNegativeMiningRounds = 10;
learnParams.balanceDatasets = true;
learnParams.debugSuffix = '1024_s40_neg';
learnParams.partNames = {'cup'};
partModels = learnModels2(conf,train_ids,train_labels,groundTruth,learnParams);


% special treatment for cup.
% partModels = learnModels3(conf,train_ids,train_labels,groundTruth,learnParams);
