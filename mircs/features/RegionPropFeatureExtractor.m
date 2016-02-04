classdef RegionPropFeatureExtractor < ShapeFeatureExtractor
    %ShapeFeatureExtractor Extracts shape features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = RegionPropFeatureExtractor(conf)
            obj = obj@ShapeFeatureExtractor(conf);
        end
        
        function x = getShapeFeatures(obj,subs,masks,regions)
            rprops = {};
            for k = 1:length(regions)
                rprops{k} = regionprops(regions{k},'Area','Eccentricity','Solidity','MajorAxisLength','MinorAxisLength',...
                    'Orientation');
                rprops{k} = rprops{k}(1);
            end
            n = numel(regions{1});
            rprops = cat(1,rprops{:});
            area_ = [rprops.Area];
            eccen_ = [rprops.Eccentricity];
            solid_ = [rprops.Solidity];
            maj_ = [rprops.MajorAxisLength];
            min_ = [rprops.MinorAxisLength];
            ori_ = [rprops.Orientation];
            
            x = [area_;area_/n;eccen_;maj_/sqrt(n);...
                min_/sqrt(n);solid_;cosd(ori_);sind(ori_)];
        end
    end
end
