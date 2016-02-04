classdef RelativeBOWFeatureExtractor < RelativeFeatureExtractor
    properties
        myBOWExtractor
    end
    
    methods
        function obj = RelativeBOWFeatureExtractor(conf)
            obj = obj@RelativeFeatureExtractor(conf)
            obj.myBOWExtractor = BOWFeatureExtractor(conf,conf.featConf);
            obj.myBOWExtractor.doPostProcess = true;
            obj.myBOWExtractor.useRectangularWindows = false;
            
        end
        function x = extractFeatures(obj,imageID,regions,pairs)
            %             rprops = struct;
            x = {};
            
            debug_ = false;
            
            d = cellfun(@(x) bwdist(x) <= 2, regions,'UniformOutput',false);
            rois = {};
            for iPair = 1:size(pairs,1)
                pair1 = pairs(iPair,1);
                pair2 = pairs(iPair,2);
                r1 = regions{pair1};
                r2 = regions{pair2};
                d1 = d{pair1};
                d2 = d{pair2};
                boundaryRegion = d1 & d2;
                [ii,jj] = find(boundaryRegion);
                if (isempty(ii))
                    warning(['no intersection between regions ' num2str(pairs(iPair,:))]);
                    continue;
                end                
                rois{end+1} = find(boundaryRegion);
            end
            if (isempty(rois))
                x = [];
            else
                x = obj.myBOWExtractor.extractFeatures(imageID,rois);
            end
        end
        % TODO - make this multiresolution / multi-angle, like you thought
        % previously.
    end
end
