classdef RelativeLayoutFeatureExtractor < RelativeFeatureExtractor
    properties
    end
    
    methods
        function obj = RelativeLayoutFeatureExtractor(conf)
            obj = obj@RelativeFeatureExtractor(conf)
  
        end
        function res = extractFeatures(obj,imageID,regions,pairs)
            %             rprops = struct;
            res = {};            
            debug_ = false;            
            rois = {};
            for iPair = 1:size(pairs,1)
                pair1 = pairs(iPair,1);
                pair2 = pairs(iPair,2);
                r1 = regions{pair1};
                r2 = regions{pair2};
                
                [y1,x1] = find(r1);
                [y2,x2] = find(r2);
                xmin = min(x1); xmax = max(x1);
                ymin = min(y1); ymax = max(y2);
                y2 = mean(y2); x2 =mean(x2);
                x_pos = 1;
                y_pos = 1;
                if (x2 > xmax)
                    x_pos = 3;
                elseif (x2 > xmin)
                    x_pos = 2;
                end
                if (y2 > ymax)
                    y_pos = 3;
                elseif (y2 > ymin)
                    y_pos = 2;
                end
                xx = zeros(3);                
                xx(y_pos,x_pos) = 1;
%                 clf; subplot(1,2,1);imagesc(r1 + 2*r2); axis image; 
%                 subplot(1,2,2); imagesc(xx); axis image;
%                  pause;
                
                res{iPair} = xx(:);
            end
            res = cat(2,res{:});
        end
        % TODO - make this multiresolution / multi-angle, like you thought
        % previously.
    end
end
