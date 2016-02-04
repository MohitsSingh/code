obj_data_dummy = struct;
t = 1158;
curPreds = L.preds{t};
curScores = L.scores{t};
curScores = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
% obtain candidates for action objects.
p = curPreds;
p(p<=3) = 0;
I = imdb.images_data{t};
I = imResample(I,[384 384]);
groundTruthMap = imResample((imdb.labels{t}>=4),size2(I),'nearest');
groundTruthMap = bwmorph(groundTruthMap,'clean');
has_gt = any(groundTruthMap(:));
props = {};
for u = 4:nLabels
    curLabels = p == u;
    curProps = regionprops(curLabels,'BoundingBox','FilledImage','PixelIdxList');
    for ttt = 1:length(curProps)
        curProps(ttt).is_gt_region = false;
    end
    props{end+1} = regionprops(curLabels,'BoundingBox','FilledImage','PixelIdxList');
end
if has_gt
    props_gt = regionprops(groundTruthMap,'BoundingBox','FilledImage','PixelIdxList');
    gt_inds = {props_gt.PixelIdxList};
    if fra_db(t).isTrain
        props{end+1} = props_gt;
    end
end
props = cat(1,props{:});

for ii = 1:length(props)
    curBox = props(ii).BoundingBox;
    curBox(3:4) = curBox(3:4)+curBox(1:2);
    s =  BoxSize(curBox);
    curBox = round(makeSquare(curBox));
    localScores = cropper(curScores,curBox);
    %     featCount = featCount + 1;
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
    obj_data_dummy(featCount) = ...
        struct('image_ind',t,'isTrain',fra_db(t).isTrain,'bbox',...
        curBox,'gt_ovp',cur_ovp,'feats',localScores);
    
end
