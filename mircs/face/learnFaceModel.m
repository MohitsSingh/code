function faceModel = learnFaceModel(conf)
imageData = initImageData;
groundTruth = struct('name',{},'sourceImage',{},'polygon','partID');

%
conf.get_full_image = true;
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
[train_ids] = getImageSet(conf,'train');
% train_labels = false(size(train_ids));
% for k = 1:length(trainData.imageIDs)
%     if (mod(k,100)==0)
%         k
%     end
%     f = find(cellfun(@any,strfind(train_ids,trainData.imageIDs{k})));
%     train_labels(f) = true;
% end
[s,is] = sort(trainData.faceScores,'descend');
groundTruth = groundTruth(is(1:100)); %TODO - note I took only the top faces


%%
[learnParams,conf] = getDefaultLearningParams(conf);
otherExtractors = learnParams.featureExtractors(4:end);
featExtractors = learnParams.featureExtractors(1:3);
for k = 1:3
    featExtractors{k}.useRectangularWindows = false;
    featExtractors{k}.doPostProcess = false;% TODO!!
end
% % % for k = 1:3
% % %     learnParams.featExtractors{k}.useRectangularWindows = false;
% % %     learnParams.featExtractors{k}.doPostProcess = true;% TODO!!
% % % end

% % % % learnParams.featureExtractors = learnParams.featureExtractors(1:3);
mbf = MultiBowFeatureExtractor(conf,featExtractors);
mbf.doPostProcess = true;
learnParams.featureExtractors = {mbf};
%%

% [learnParams,conf] = getDefaultLearningParams(conf);
learnParams.partNames = {'face'};
% learnParams.featureExtractors = learnParams.featureExtractors(1:3);
% for k = 1:length(learnParams.featureExtractors)
%     learnParams.featureExtractors{k}.useRectangularWindows = false;
% end
learnParams.debugSuffix = 'bow_face_s';
learnParams.class_name = 'generic_p';
learnParams.useRealGTSegments = false;
learnParams.doHardNegativeMining = true;

nonPersonIds = getNonPersonIds(conf.VOCopts);
for k = 1:length(nonPersonIds)
    nonPersonIds{k} = [nonPersonIds{k} '.jpg'];
end
learnParams.negImages = nonPersonIds;
faceModel = learnModels2(conf,train_ids,[],groundTruth,learnParams);
end