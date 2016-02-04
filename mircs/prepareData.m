function fra_db = prepareData(conf,fra_db,params)
roiParams = params.roiParams;
tt = ticStatus('prepareData-->loading ground truth',.5,.5,false);
for t = 1:length(fra_db)
%     t
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
    fra_db(t).I = I;
    fra_db(t).roiBox = roiBox;
    fra_db(t).gt_obj = [];
    iObj = find(strcmp({rois.name},'obj'));
    
    fra_db(t).isValid = true;
    if none(iObj)
        fra_db(t).isValid = false;
        continue
    end
    fra_db(t).gt_obj = {rois(iObj).poly};
    iHand = find(strcmp({rois.name},'hand'));
    hands_to_keep = fra_db(t).hands_to_keep;
    if (any(iHand))
        fra_db(t).gt_hand = cat(1,rois(iHand(hands_to_keep)).bbox);
    end
    [xy,goods] = loadKeypointsGroundTruth(fra_db(t),params.requiredKeypoints);
    %     confidences = xy(:,3);
    xy = xy(:,1:2);
    xy = xy-repmat(roiBox(1:2),size(xy,1),1);
    xy = xy*scaleFactor;
    fra_db(t).face_landmarks = struct('xy',xy,'valids',goods);
    tocStatus(tt,t/length(fra_db));
end