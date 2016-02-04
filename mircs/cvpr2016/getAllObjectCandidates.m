function training_data = getAllObjectCandidates(LL, candidateParams , imdb, fra_db,u)
training_data = struct('imgInd',{},...
    'bbox',{},...
    'bbox_orig',{},...
    'isTrain',{},...
    'PixelIdxList',{},...
    'ovp_region',{},...
    'ovp_big_box',{},...
    'ovp_orig_box',{},...
    'is_gt_region',{},...
    'gt_region_labels',{});
%     'label',{},...
% 'feats_coarse',{},...
%     'feats_fine',{},...
nSamples = 0;
for t = u%944:1:length(fra_db);
    
    %     if ~fra_db(t).isTrain,continue,end
    %         if fra_db(t).classID~=5,continue,end
    t
    if (mod(t,15)==0)
        t
    end
    coarse = LL.scores_coarse{t};
    fine = LL.scores_fine{t};
    L.scores_coarse = coarse;
    L.scores_fine = fine;
    I = imdb.images_data{t};
    %     I_rect = imdb.rects(t,:);
    props = getObjectCandidates(L,imdb,fra_db,t,candidateParams,false);
    curBoxesOrig = round(cat(1,props.bbox));
    curBoxesBig = round(inflatebbox(cat(1,props.bbox),size(I,1)/3,'both',true));
    regions = propsToRegions(props,size2(I));
    cur_labels = [props.is_gt_region];
    f_gt = cur_labels>0;
    %     f_other = cur_labels==0;
    boxes_gt_big = curBoxesBig(f_gt,:);
    boxes_gt = curBoxesOrig(f_gt,:);
    regions_gt = regions(f_gt);
    %     have_labels = cur_labels(f_gt);
    overlaps_region = regionsOverlap3(regions,regions_gt);
    % make sure there's a proposal for each ground truth region.
    [ overlaps_big ,ints] = boxesOverlap( curBoxesBig,boxes_gt_big);
    [ overlaps_orig ,ints] = boxesOverlap( curBoxesOrig,boxes_gt);
    %     if not(any(have_labels>2))
    %         overlaps_region(:,3) = 0;
    %         overlaps_big(:,3) = 0;
    %     end
    
    % set the label of each box as that of the maximally overlapping
    % ground truth box but only if the overlap is > .5
    
    [v,iv] = max(overlaps_orig,[],2);
    
    %     if (any(v(f_gt))<.5)
    %         ff = find(v(f_gt<5));
    %         disp(ff);
    %         z_dummy = 0;
    %     end
    cur_labels = cur_labels(f_gt);
    %     boxLabels = cur_labels(iv);
    %     boxLabels(v < .5) = -1;
    %     training_data(t)
    if ~fra_db(t).isTrain
        %         boxesToRemove = v < 1 & v > .8; % no need for near duplicates.
        %     else
        boxesToRemove = [props.is_gt_region] > 0;
        %         boxesToRemove = v==1; % remove ground truth regions from
        %         test data.
        %         boxesToRemove = [];
        curBoxesBig(boxesToRemove,:) = [];
        curBoxesOrig(boxesToRemove,:) = [];
        %     boxLabels(boxesToRemove) = [];
        props(boxesToRemove) = [];
        overlaps_region(boxesToRemove,:) = [];
        overlaps_big(boxesToRemove,:) = [];
        overlaps_orig(boxesToRemove,:) = [];
        f_gt(boxesToRemove) = [];
        regions(boxesToRemove) = [];
        if any(max(overlaps_orig,[],1)>.5)
%             [double(cur_labels);max(overlaps_orig,[],1)]            
            [z,iz] = min(max(overlaps_orig));
%             cur_labels(iz)
% % % %             clf; imagesc2(I);drawnow            
% % % %             displayRegions(I,regions,overlaps_region(:,iz),'maxRegions',3,'delay',.01);
%             dpc
            sfsdf=1;
        end
    end
    
    
    
    %     I = imdb.images_data{t};
    for n = 1:length(props)
        nSamples = nSamples+1;
        curBox = curBoxesBig(n,:);
        feats_coarse = cropper(coarse,curBox);
        feats_coarse = imResample(feats_coarse,[7,7],'bilinear');
        feats_fine = cropper(fine,curBox);
        feats_fine = imResample(feats_fine,[7,7],'bilinear');
        training_data(nSamples) = struct('imgInd',t,...
            'bbox',curBoxesBig(n,:),...
            'bbox_orig',curBoxesOrig(n,:),...
            'isTrain',fra_db(t).isTrain,...
            'PixelIdxList',props(n).PixelIdxList,...
            'ovp_region',overlaps_region(n,:),...
            'ovp_big_box',overlaps_big(n,:),...
            'ovp_orig_box',overlaps_orig(n,:),...
            'is_gt_region',f_gt(n),...
            'gt_region_labels',cur_labels);
        %             'feats_coarse',feats_coarse,...
        %             'feats_fine',feats_fine,...
        
        %             'label',boxLabels(n),...
        %                 clf; imagesc2(I); plotBoxes(curBoxes(n,:)); title(num2str(boxLabels(n)));
        %                 dpc
    end
end
