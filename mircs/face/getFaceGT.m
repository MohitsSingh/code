function groundTruth = getFaceGT(conf)
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
    groundTruth(k).partID = 5;
    groundTruth(k).objectID = k;
    
    groundTruth(k).occlusion = [];
    groundTruth(k).representativeness = [];
    groundTruth(k).uncertainty = [];
    groundTruth(k).deleted = [];
    groundTruth(k).verified = [];
    groundTruth(k).date = [];
    groundTruth(k).sourceAnnotation = [];
    
    groundTruth(k).objectParts = [];
    groundTruth(k).comment = [];
    
end
% find the images not including these top faces...
% [train_ids] = getImageSet(conf,'train');
% train_labels = false(size(train_ids));
% for k = 1:length(trainData.imageIDs)
%     if (mod(k,100)==0)
%         k
%     end
%     f = find(cellfun(@any,strfind(train_ids,trainData.imageIDs{k})));
%     train_labels(f) = true;
% end
% [s,is] = sort(trainData.faceScores,'descend');
% groundTruth = groundTruth(is(1:100)); %TODO - note I took only the top faces
