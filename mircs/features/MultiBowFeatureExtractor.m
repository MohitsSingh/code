classdef MultiBowFeatureExtractor < CompositeFeatureExtractor
    %BowFeatureExtractor Extracts bag-of-words features from regions or windows.
    %   Detailed explanation goes here
    
    methods
        function obj = MultiBowFeatureExtractor(conf,extractors)
            obj = obj@CompositeFeatureExtractor(conf,extractors);
        end
        function x = extractFeatures(obj,imageID,roi,varargin)
            x = obj.extractFeaturesHelper(imageID,roi,varargin{:});
            
            %             extractFeatures@CompositeFeatureExtractor(imageID,roi,varargin{:});
            
            if (obj.doPostProcess)
                x = bsxfun(@rdivide,x,sum(x,1));
                x = (vl_homkermap(full(x), 1, 'kchi2', 'gamma', 1));
            else
                x = bsxfun(@rdivide,x,sum(x.^2).^.5);
            end
        end
    end
end