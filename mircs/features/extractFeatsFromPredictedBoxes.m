
function [box_data,image_data] = extractFeatsFromPredictedBoxes(conf,fra_db,img_sel,params,bboxPredictor)

% boxes_data = struct('imgIndex',{},'bbox',{},'feats',{},'ovp',{});
% image_data = {};

image_data = {};
box_data =  {};

f_sel = find(img_sel);
trainingDataDir = '~/storage/s40/boxFeatsData';ensuredir(trainingDataDir);
% trainingDataDir = '~/data/boxFeatsData1';ensuredir(trainingDataDir);
% pp = randperm(length(f_sel));
pp = 1:length(f_sel);
nBoxesPerImage = 50;
for iu =1:length(f_sel)
    u = pp(iu);
    u
    imgData = fra_db(f_sel(u));
    curResPath = j2m(trainingDataDir,imgData);
    fileExists = false;
    if (exist(curResPath,'file'))
        try
            load(curResPath);
            imgIndices = [boxFeats.imgIndex];
            sel_ = imgIndices==u;
            if (~all(sel_))
                nnz(sel_)/length(sel_)
                boxFeats = boxFeats(sel_);
                save(curResPath,'imageFeats','boxFeats');
            end
            fileExists = true;
        catch e
            fileExists = false;
        end
    end
    if (~fileExists)
        
        featPath = j2m(params.featsDir,imgData);
        L = load(featPath);
        I = L.imageFeats.I;
        [labels,features,ovps,is_gt_region] = collectFeaturesFromImg(conf,imgData,params);
        % remove duplicate features, boxes
        features = features';
        [features,i_subset] = unique(features,'rows');
        labels = labels(i_subset);
        ovps = ovps(i_subset);
        is_gt_region = is_gt_region(i_subset);
        r = adaBoostApply(features,bboxPredictor,[],[],8);
        boxes = cat(1,L.regionFeats(i_subset).bbox);
        [v,iv] = sort(r,'descend');
        boxes = boxes(iv(1:5),:);
        
        imageFeats = struct('global',L.imageFeats.global_feats,'face',L.imageFeats.face_feats.global,...
            'mouth',L.imageFeats.face_feats.mouth);
        imgData = switchToGroundTruth(imgData);
        [rois,roiBox,~,scaleFactor,roiParams] = get_rois_fra(conf,imgData,params.roiParams);
        iObj = find(strncmpi('obj',{rois.name},3));
        obj_box = [0 0 0 0];
        
        if (any(iObj))
            best_ovp = 0;bestBox = 0;
            objBoxes = cat(1,rois(iObj).bbox);
            imgBox = [1 1 fliplr(size2(I))];
            [obj_ovps,obj_ints] = boxesOverlap(objBoxes,imgBox);
            [~,~,s] = BoxSize(objBoxes);
            [~,ii] = max(obj_ints./s);
            objBoxes = objBoxes(ii,:);
            obj_box = round(BoxIntersection(objBoxes,imgBox));
        end
                                
        subImgs = multiCrop2(I, boxes);
        x_17 = extractDNNFeats(subImgs,params.features.nn_net);
        boxFeats = struct('imgIndex',{},'bbox',{},'ovp',{},'bb_ovp',{},'feats',{});
        for t = 1:5
            boxFeats(t).imgIndex = u;
            boxFeats(t).bbox = boxes(t,:);
            boxFeats(t).ovp = ovps(iv(t));
            boxFeats(t).bb_ovp = boxesOverlap(obj_box,boxes(t,:));
            boxFeats(t).feats = x_17(:,t);
        end
        
        save(curResPath,'imageFeats','boxFeats');
    end
    image_data{end+1} = imageFeats;
    box_data{end+1} = boxFeats;
end