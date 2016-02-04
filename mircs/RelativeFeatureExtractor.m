classdef RelativeFeatureExtractor < handle
    %%
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        conf
        doPostProcess = false
    end
    
    methods
        function obj = RelativeFeatureExtractor(conf)
            obj.conf = conf;
        end             
            % extract features for all pairs of rois in the graph...
            
%             x = {};
%             imageData = obj.preprocessImage(imageID,rois);
%             for iPair = 1:length(ii)
%                 x{iPair} = obj.extractFeaturesHelper(imageData,rois,ii(iPair),jj(jPair));
%             end
%            x = cat(2,x{:});
                
    end
    
    methods (Abstract)        
        x = extractFeatures(obj,imageID,rois,pairs)
    end
end

