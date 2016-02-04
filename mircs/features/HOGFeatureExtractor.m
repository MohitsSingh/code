classdef HOGFeatureExtractor < ShapeFeatureExtractor
    %ShapeFeatureExtractor Extracts shape features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = HOGFeatureExtractor(conf)
            obj = obj@ShapeFeatureExtractor(conf);
        end
        
        function x = getShapeFeatures(obj,subs,masks,regions)
            x = {};
            for k = 1:length(subs)
                x{k} = col(vl_hog(subs{k},8));
            end
            x = double(cat(2,x{:}));
        end
    end
end
