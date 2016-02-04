conf.get_full_image = true;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
[learnParams,conf] = getDefaultLearningParams(conf);
learnParams.partNames = partNames;
learnParams.debugSuffix = 'debug_f';
learnParams.doHardNegativeMining = false;
learnParams_binary = learnParams;

% RelativeGeometryFeatureExtractor gfe;
% gfe.extractors = {RelativeLayoutFeatureExtractor(conf),RelativeShapeFeatureExtractor(conf)};
    
learnParams_binary.featureExtractors = {RelativeLayoutFeatureExtractor(conf)};
learnParams_binary.featureExtractors{end+1} = RelativeShapeFeatureExtractor(conf);
learnParams_binary.featureExtractors{end+1} = RelativeGeometryFeatureExtractor(conf);
% learnParams_binary.featureExtractors{end+1} = RelativeBOWFeatureExtractor(conf);

binaryModels = learnBinaryModels(conf,train_ids,train_labels,groundTruth,learnParams_binary);