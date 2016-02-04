classdef MaskShapeFeatureExtractor < ShapeFeatureExtractor
    %ShapeFeatureExtractor Extracts shape features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        
    end
    
    methods
        function obj = MaskShapeFeatureExtractor(conf)
            obj = obj@ShapeFeatureExtractor(conf);
        end
        
        function x = getShapeFeatures(obj,subs,masks,regions)
            x = {};
            for k = 1:length(masks)
                [c] = getLogPolarShape(masks{k});

%                 [c,c_vis] = getLogPolarShape(masks{k});
%                 
%                 clf;subplot(1,2,1);imshow(masks{k});
%                 subplot(1,2,2);
%                 imshow(c_vis);
%                 title(num2str(nnz(masks{k})));
%                 pause;
                x{k} = c(:);
                if (obj.doPostProcess)
                    x{k} = x{k}/(sum(x{k}.^2)^.5); % l2 normalize
                end
                
            end
            x = cat(2,x{:});
        end
    end
end