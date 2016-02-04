function [all_labels, all_features,all_ovps,is_gt_region,orig_inds] = collectFeatures(featStruct,featureParams)
all_labels = {};
all_features = {};
all_ovps = {};
is_gt_region = {};
orig_inds = {};
for t = 1:length(featStruct)
    currentFeatData = featStruct(t);
    if (currentFeatData.imageFeats.valid)
        all_ovps{end+1} = [currentFeatData.regionFeats.gt_ovp];
        all_labels{end+1} = [currentFeatData.regionFeats.label];
        is_gt_region{end+1} = [currentFeatData.regionFeats.is_gt_region];
        currentFeatures = {};
        if (featureParams.getSimpleShape)
            for u = 1:length(currentFeatData.regionFeats)
                currentFeatData.regionFeats(u).simpleShape = currentFeatData.regionFeats(u).simpleShape(:);
            end
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.simpleShape);
        end
%         if (featureParams.getAttributeFeats)
%             currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.attributeFeats);
%         end
        if (featureParams.getGeometry)
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.geometricFeats).^.5;
        end
        if (featureParams.getGeometryLogPolar)
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.geometric_feats_log_polar)>0;
        end
        %                     assert(size(currentFeatures{end},1)==42)
        if (featureParams.getPixelwiseProbs)
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.meanSaliency);
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.meanLocationProb);
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.meanShapeProb);
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.meanGlobalSaliency);
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.meanGlobalBDSaliency);
            
        end
        if (featureParams.getShape)
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.shapeFeats);
        end
        if (featureParams.getAppearance)
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.appearance);
        end
        
        if featureParams.getBoxFeats
            currentFeatures{end+1} = cat(1,currentFeatData.regionFeats.boxFeats)';
        end
        
        if featureParams.getHOGShape
            for iFeat = 1:length(currentFeatData.regionFeats)
                currentFeatData.regionFeats(iFeat).HOGShape = currentFeatData.regionFeats(iFeat).HOGShape(:);
            end
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.HOGShape);
        end
        
        if featureParams.getAppearanceDNN            
            currentFeatures{end+1} = cat(2,currentFeatData.regionFeats.dnn_feats);
        end
      
        
        all_features{end+1} = cat(1,currentFeatures{:});
        orig_inds{end+1} = t*ones(size(all_features{end},2),1);
    end
end


all_labels = cat(2,all_labels{:});
is_gt_region = cat(2,is_gt_region{:});
all_features = single(cat(2,all_features{:}));
all_ovps = cat(2,all_ovps{:});
