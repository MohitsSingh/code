classdef MaskHOGFeatureExtractor < ShapeFeatureExtractor
    %ShapeFeatureExtractor Extracts shape features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = MaskHOGFeatureExtractor(conf)
            obj = obj@ShapeFeatureExtractor(conf);
        end
        
        function x = getShapeFeatures(obj,subs,masks,regions)
            x = {};
            for k = 1:length(masks)
                x{k} = col(vl_hog(masks{k},8));
            end
            x = cat(2,x{:});
        end
    end
end