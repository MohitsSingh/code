function [moreData] = extract_all_features_lite(conf,imgData,params,moreData,selected_regions)
%Aggregates multiple feature types for images. Some features may have been
% precomputed

% obtain:
% 1. image
% 2. facial keypoints
% 3. segmentation
% 4. saliency
% 5. object mask
% 6. shape probability prediction
% 7. line segment features (to server as additional candidates.

feats = struct('label',{},'mask',{},'geometricFeats',{},'shapeFeats',{},'meanSaliency',{},'meanProb',{});
if (imgData.valid==0)
    feats = [];
    moreData =[];
    selected_regions = [];
    return;
end
if (isfield(params,'prevStageDir') &&~isempty(params.prevStageDir))
    load(j2m(params.prevStageDir,imgData));
    selected_regions = toKeep;
end

if (isfield(params,'get_gt_regions'))
    params.get_gt_regions = true;
end
isExternalDB = false;
if (isfield(params,'externalDB') && params.externalDB)
    is_in_fra_db = -1;
    isExternalDB=  true;
else
    is_in_fra_db = imgData.indInFraDB~=-1;
end

imgData = switchToGroundTruth(imgData);

[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,imgData,params.roiParams);
moreData.I = I;


[I_orig,I_rect] = getImage(conf,imgData);
% facial keypoints

iMouth = find(strncmpi('mouth',{rois.name},5));
moreData.roiMouth = inflatebbox(rois(iMouth).bbox,[60 40],'both',true);
if (params.skipCalculation)
    showStuff();
    return;
end
net = params.features.dnn_net;
[moreData.face_feats.global_17,moreData.face_feats.global_19] = extractDNNFeats(I,net);
[moreData.face_feats.mouth_17,moreData.face_feats.mouth_19] = extractDNNFeats(cropper(I,round(makeSquare(moreData.roiMouth,true))),net);
[moreData.global_feats_17,moreData.global_feats_19] = extractDNNFeats(I_orig,net);
get_full_image = false;
I_crop = getImage(conf,imgData,get_full_image);
[moreData.global_feats_crop_17,moreData.global_feats_crop_19] = extractDNNFeats(I_crop,net);
