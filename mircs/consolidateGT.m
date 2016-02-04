function groundTruth = consolidateGT(conf,trainOrTest,override)
if nargin < 3
    override = false;
end

gtPath = fullfile(conf.cachedir,['gt_' trainOrTest '.mat']);
if (~exist(gtPath,'file') || override)
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,trainOrTest);
    % annotateFaces(conf,train_ids(all_train_labels==conf.class_enum.BLOWING_BUBBLES),[],'mouth');
    train_mouths = annotateFaces(conf,train_ids(train_labels),[],'mouth');
    % [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
    % annotateFaces(conf,test_ids(test_labels),[],'mouth');    
    gtMouth = convertToGroundTruth(train_mouths,'mouth');
    [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
    groundTruth = [groundTruth,gtMouth];
    save(gtPath,'groundTruth');
else
    load(gtPath);
end

