function [res,normalizations] = fra_extract_feats_helper(conf,curImageData,roiParams,featureExtractor)
iFeat = 0;
normalizations = {'none','Normalized','SquareRoot','Improved'};
for flip = [0 1]
    [rois,subRect,I] = get_rois_fra(conf,curImageData,roiParams);
    if (flip)
        I = flip_image(I);
    end
    roiMasks = {};
    for iRoi = 1:length(rois)
        iFeat = iFeat+1;
        curRoi = rois(iRoi);
        curBox = curRoi.bbox;
        if (flip)
            curBox = flip_box(curBox,size2(I));
        end
        
        res(iFeat).bbox = curBox;
        curMask = poly2mask2(round(curBox),size2(I));
        res(iFeat).isTrain = curImageData.isTrain;
        res(iFeat).srcImageIndex = curImageData.imgIndex;
        res(iFeat).type = curRoi.id;
        res(iFeat).name = curRoi.name;
        res(iFeat).is_gt_location = true;
        res(iFeat).flipped = flip;
        %                 clf; displayRegions(I,curMask); pause; continue;
        
        % loop over normalization schemes
        curFeats = [];
        for iNormalize = 1:length(normalizations)
            curFeats = [curFeats,featureExtractor.extractFeatures(I,curMask,'normalization',normalizations{iNormalize})];
        end
        
        res(iFeat).feat = curFeats;
        res(iFeat).isValid = none(isinf(curFeats) | isnan(curFeats)) & ~isempty(curFeats);
    end
end