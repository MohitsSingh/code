function props = getObjectCandidates(L,imdb,fra_db,t,params,debugging)
if nargin < 6
    debugging = false;
end
% curPreds = L.preds{t};

I = imdb.images_data{t};
I = im2single(I);
groundTruthMap = imdb.labels{t};
has_gt = any(groundTruthMap(:));
curScores = (L.scores_coarse+L.scores_fine)/2;
[~,curPreds] = max(curScores,[],3);
% obtain candidates for action objects.
props = getRegionProposals(curPreds,curScores,I,params);
% curScores = L.scores_fine;
% [~,curPreds] = max(curScores,[],3);
% obtain candidates for action objects.
% props = [props;getRegionProposals(curPreds,curScores,I,params)];

%[D R] = DT(1-M);
% x2(M>.5)
% if length(props)>0
%     props([props.Area]<10) = [];
% end
% displayRegions(I,propsToRegions(props,size2(I)),[props.MeanIntensity]);
M = max(curScores(:,:,2:end),[],3);
if has_gt
    props_gt = regionprops(groundTruthMap,groundTruthMap,'Area','MaxIntensity','MeanIntensity','BoundingBox','FilledImage','PixelIdxList');
    toRemove = isnan([props_gt.MeanIntensity]);
    props_gt(toRemove) = [];
    for u = 1:length(props_gt)
        props_gt(u).is_gt_region = props_gt(u).MaxIntensity;
    end
    
    % fill in the mean intensity.
    props_gt_1 = regionprops(groundTruthMap,M,'Area','MaxIntensity','MeanIntensity');
    props_gt_1(toRemove) = [];
    for u = 1:length(props_gt)
        props_gt(u).MaxIntensity = props_gt_1(u).MaxIntensity;
        props_gt(u).MeanIntensity = props_gt_1(u).MeanIntensity;
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
