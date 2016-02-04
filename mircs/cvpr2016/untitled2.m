
s = res_train(1).classifier_data.w(1:end-1)'*feats_r1;

[r,ir] = sort(s,'descend');
image_visited = zeros(size(fra_db));

showEachImageOnce = true;
debug_factor = 0;
showTrainingImages = true;
showTestingImages = true;
showAnything = false;
[r,ir] = sort(s,'descend');
% curValids = true(size(fra_db));

chosenScores=  {};
for it = 1:1:length(s)
    r(it)
    k = ir(it);
    imgInd = all_img_inds(k);
    if isTrain(imgInd)
        if ~showTrainingImages,continue,end
    else
        if ~showTestingImages,continue,end
    end
%     if showTrainingImages && ~isTrain(imgInd),continue,end
    if ~isempty(img_sel) && imgInd~=img_sel,continue,end
    if showEachImageOnce
        if (image_visited(imgInd)),continue,end
    end
    image_visited(imgInd)=1;
    nImagesVisited = nnz(image_visited);
    nImagesVisited                
%     if isTrain(imgInd),continue,end
    I = imdb.images_data{imgInd};
    patch_of_img{imgInd} = cropper(I,all_boxes(k,:));
    patch_of_img_bigger{imgInd} = cropper(I,round(inflatebbox(all_boxes(k,:),2,'both',false)));
    chosenFeats{imgInd} = feats_r1(:,k);
    chosenScores{imgInd} = r(it);
        if debug_factor ~=0 && mod(nImagesVisited,debug_factor)~=0,continue,end
    if ~showAnything,continue,end
    figure(10); clf;imagesc2(I);
    plotBoxes(all_boxes(k,:));
%     showPredsHelper2(fra_db,imdb,imgInd);
    dpc
end