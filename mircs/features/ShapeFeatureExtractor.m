
classdef ShapeFeatureExtractor < FeatureExtractor
    %ShapeFeatureExtractor Extracts shape features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        bowConf
        absoluteFrame = false
    end
    
    methods
        function obj = ShapeFeatureExtractor(conf)
            obj = obj@FeatureExtractor(conf);
        end
        
        function x = extractFeaturesHelper(obj,currentID,regions)
            I = getImage(obj.conf,currentID);
            % get bounding boxes for all regions
            I =  im2single(I);
            subs = {};
            masks = {};
            for iRegion = 1:length(regions)
                if (obj.absoluteFrame)
                    subI = I;
                    x = regions{iRegion};
                else
                    [subI,x] = obj.getShapeAndSubImage(I,regions{iRegion});
                end
                subI = imresize(subI,[80 80],'bilinear');
                curMask = imresize(single(x),[80 80],'bilinear');
                subs{iRegion} = subI;
                masks{iRegion} = curMask;
            end
            x = obj.getShapeFeatures(subs,masks,regions);
        end
        
        function [I,x,mask] = getShapeAndSubImage(obj,I,x)
            [ii,jj] = find(x);
            xmin = min(jj); xmax = max(jj);
            ymin = min(ii); ymax = max(ii);
            rect = [ymin xmin ymax xmax];
            rect = round(makeSquare(rect));            
            startLocs = [rect(1:2) 1];
            endLocs = [rect(3:4) size(I,3)];
            I = arrayCrop(I,startLocs,endLocs);
            x = arrayCrop(x,startLocs(1:2),endLocs(1:2));
        end
    end
    
    methods (Abstract)
        x = getShapeFeatures(obj,subs,masks,regions);
    end
end
