function props = getObjectCandidates(L,imdb,fra_db,t,params,debugging)
if nargin < 6
    debugging = false;
end
curPreds = L.preds{t};
curScores = L.scores{t};
softMaxScores = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));

% obj_data = struct('image_ind',{},'isTrain',{},'bbox',...
%     {},'gt_ovp',{},'feats',{},'is_gt_region',{},'PixelIdxList',{},'area',{},'meanIntensity',{});

I = imdb.images_data{t};
I = im2single(I);

% obj_data = struct;
I = imResample(I,[384 384]);
% x2(max(softMaxScores(:,:,2:end),[],3))
groundTruthMap = imResample((imdb.labels{t}>=4),size2(I),'nearest');
groundTruthMap = bwmorph(groundTruthMap,'clean');
has_gt = any(groundTruthMap(:));
% obtain candidates for action objects.
props = getRegionProposals(curPreds,softMaxScores,I,params);
%props([props.Area]<100) = [];
% displayRegions(I,propsToRegions(props,size2(I)));
M = max(softMaxScores(:,:,2:end),[],3);
if has_gt
    props_gt = regionprops(groundTruthMap,M,'Area','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
    for u = 1:length(props_gt)
        props_gt(u).is_gt_region = true;
    end
    %     gt_inds = {props_gt.PixelIdxList};
    %     if fra_db(t).isTrain
    props = [props;props_gt];
    %     end
end

for t = 1:length(props)
    curBox = props(t).BoundingBox;
    curBox(3:4) = curBox(3:4)+curBox(1:2);
    props(t).bbox = curBox;
    props(t).isTrain = fra_db(t).isTrain;
end


return;
%%

% props = cat(1,props{:});
featCount = 0;
if debugging
    showPredictions(255*single(I),curPreds,softMaxScores,L.labels,1);
    dpc
    %     return
    %     figure(2); clf; imagesc2(groundTruthMap);
    %       bb = cat(1,props.BoundingBox);
    %     bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
end
for ii = 1:length(props)
    curBox = props(ii).BoundingBox;
    curBox(3:4) = curBox(3:4)+curBox(1:2);
    s =  BoxSize(curBox);
    curBox = round(inflatebbox(curBox,size(I,1)/3,'both',true));
    if debugging
        figure(4);
        clf; imagesc2(I);
        if props(ii).is_gt_region
            plotBoxes(curBox,'r-','LineWidth',2);
        else
            plotBoxes(curBox,'g-','LineWidth',2);
        end
        showPredictions(single(cropper(I,curBox)),...
            cropper(curPreds,curBox),...
            cropper(softMaxScores,curBox),L.labels,1);
    end
    %     dpc
    localScores = cropper(softMaxScores,curBox);
    localScores = imResample(localScores,[5,5],'bilinear');
    featCount = featCount + 1;
    my_inds = props(ii).PixelIdxList;
    if has_gt
        nGT = length(gt_inds);
        cur_ovp =0;
        for igt = 1:nGT
            cur_gt_inds = gt_inds{igt};
            cur_ovp = max(length(intersect(my_inds,cur_gt_inds))/length(union(my_inds,cur_gt_inds)),cur_ovp);
        end
    else
        cur_ovp = 0;
    end
    obj_data(featCount) = ...
        struct('image_ind',t,'isTrain',fra_db(t).isTrain,'bbox',...
        curBox,'gt_ovp',cur_ovp,'feats',localScores,'is_gt_region',props(ii).is_gt_region,...
        'PixelIdxList',props(ii).PixelIdxList,'area',props(ii).Area,'meanIntensity',props(ii).MeanIntensity);
end
