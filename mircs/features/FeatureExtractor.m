classdef FeatureExtractor < handle
    %FeatureExtractor Extracts features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        conf
        doPostProcess = true
        isNormalized = false;
        normalizer = struct('mean',{},'std',{});
    end
    
    methods
        function obj = FeatureExtractor(conf)
            obj.conf = conf;
        end
        
        function x = extractFeatures(obj,imageID,roi,varargin)
            if iscell(imageID) && length(imageID)==1
                imageID = imageID{1};
            end
            if (nargin < 3)
                roi = [];
            end
            if (~isempty(roi))
                %             if (nargin > 2)
                if (~iscell(roi))
                    roi = {roi};
                end
            elseif (~ischar(imageID))
                roi = {true(size2(imageID))};
            else
                roi = getRegions(obj.conf,imageID,false);
            end
            x = obj.extractFeaturesHelper(imageID,roi,varargin{:});
            if (obj.isNormalized)
                x = obj.normalizeFeatures(x);
            end
        end
        function x = description(obj)
            x = class(obj);
        end
        
        function fixNormalization(obj,x)
            % find std and variance of each channel.
            obj.isNormalized = true;
            x_mean = mean(x,2);
            x_std = std(x,0,2);
            
            % ignore values of 0 std by borrowing from neighboring values.
            bads = find(x_std == 0);
            for iBad = 1:length(bads)
                r = max(1,bads(iBad)-10):min(length(x_mean),bads(iBad)+10);
                vals = x_std(r);
                v = mean(vals(vals ~= 0));
                x_std(bads(iBad)) = v;
                vals = x_mean(r);
                v = mean(vals(vals ~= 0));
                x_mean(bads(iBad)) = v;
            end
            
            
            obj.normalizer(1).mean = x_mean;%mean(x,2);
            obj.normalizer(1).std = x_std;%std(x,0,2);
            
        end
        
        function x = normalizeFeatures(obj,x)
            x = bsxfun(@minus,x,obj.normalizer.mean);
            x = bsxfun(@rdivide,x,obj.normalizer.std);
        end
    end
    methods (Abstract)
        x = extractFeaturesHelper(obj,imageID, roi, varargin);
    end
end
