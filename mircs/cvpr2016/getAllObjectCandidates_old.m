function training_data = getAllObjectCandidates(LL, candidateParams , imdb, fra_db)
training_data = struct('imgInd',{},'feats_coarse',{},'feats_fine',{},'label',{},'bbox',{},'isTrain',{},'PixelIdxList',{});
nSamples = 0;

for t = 1:1:length(fra_db);
    if ~fra_db(t).isTrain,continue,end
    %         if fra_db(t).classID~=5,continue,end
    t
    if (mod(t,50)==0)
        t
    end
    coarse = LL.scores_coarse{t};
    fine = LL.scores_fine{t};
    L.scores_coarse = coarse;
    L.scores_fine = fine;
    props = getObjectCandidates(L,imdb,fra_db,t,candidateParams,false);
    I = imdb.images_data{t};
    %curBoxes = round(inflatebbox(makeSquare(cat(1,props.bbox),true),size(I,1)/3,'both',false));
    
%     {props.PixelIdxList}
    
%     x2(I);plotBoxes(props(1).bbox)
        
    curBoxes = round(inflatebbox(cat(1,props.bbox),size(I,1)/3,'both',true));
%     curBoxes = round(cat(1,props.bbox));
    cur_labels = [props.is_gt_region];            
    f_gt = cur_labels>0;
    f_other = cur_labels==0;
    boxes_gt = curBoxes(f_gt,:);
    boxes_other = curBoxes(f_other,:);
    %     x2(I);plotBoxes(curBoxes)
    regions = propsToRegions(props,size2(I));
    regions_gt = regions(f_gt);
    overlaps = regionsOverlap3(regions,regions_gt);
    
    %%%[ overlaps ,ints] = boxesOverlap( curBoxes,boxes_gt);                
    % set the label of each box as that of the maximally overlapping
    % ground truth box but only if the overlap is > .5
    [v,iv] = max(overlaps,[],2);
    
    displayRegions(I,regions,v)
    
    cur_labels = cur_labels(f_gt);
    boxLabels = cur_labels(iv);
    boxLabels(v < .5) = -1;
    if fra_db(t).isTrain
        boxesToRemove = v < 1 & v > .8; % no need for near duplicates.
    else
        boxesToRemove = [props.is_gt_region] > 0;
        %         boxesToRemove = v==1; % remove ground truth regions.
    end
    curBoxes(boxesToRemove,:) = [];
    boxLabels(boxesToRemove) = [];
    props(boxesToRemove) = [];
    
    I = imdb.images_data{t};
    for n = 1:length(props)
        nSamples = nSamples+1;
        curBox = curBoxes(n,:);
        feats_coarse = cropper(coarse,curBox);
        feats_coarse = imResample(feats_coarse,[7,7],'bilinear');
        feats_fine = cropper(fine,curBox);
        feats_fine = imResample(feats_fine,[7,7],'bilinear');
        training_data(nSamples) = struct('imgInd',t,...
            'feats_coarse',feats_coarse,...
            'feats_fine',feats_fine,...
            'label',boxLabels(n),...
            'bbox',curBoxes(n,:),...
            'isTrain',fra_db(t).isTrain,...
            'PixelIdxList',props(n).PixelIdxList);
        %                 clf; imagesc2(I); plotBoxes(curBoxes(n,:)); title(num2str(boxLabels(n)));
        %                 dpc
    end
end
