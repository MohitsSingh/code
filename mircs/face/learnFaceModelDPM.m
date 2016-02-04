function faceModel = learnFaceModel(conf)
imageData = initImageData;
groundTruth = struct('name',{},'sourceImage',{},'polygon','partID');

%
trainData = imageData.train;
for k = 1:length(trainData.imageIDs)
    k
    groundTruth(k).name = 'face';
    groundTruth(k).sourceImage = trainData.imageIDs{k};
    pts = box2Pts(trainData.faceBoxes(k,:));
    groundTruth(k).polygon.x = pts(:,1);
    groundTruth(k).polygon.y = pts(:,2);
    groundTruth(k).partID = 1;
end
% find the images not including these top faces...

[s,is] = sort(trainData.faceScores,'descend');
groundTruth = groundTruth(is(1:100)); %TODO - note I took only the top faces
n = getNonPersonIds(conf.VOCopts);
faceModel = learnModelsDPM(conf,n,false(size(n)),groundTruth,{'dpm_face'},n)

% learnModelsDPM
% [learnParams,conf] = getDefaultLearningParams(conf);
% learnParams.partNames = {'face'};
% learnParams.featureExtractors = learnParams.featureExtractors(1:3);
% learnParams.debugSuffix = 'bow_face';
% learnParams.class_name = 'generic';
% learnParams.doHardNegativeMining = false;
% faceModel = learnModels2(conf,train_ids,train_labels,groundTruth,learnParams);
end