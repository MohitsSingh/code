classdef RelativeLayoutFeatureExtractor < RelativeFeatureExtractor
    properties
        
    end
    
    methods
        function obj = RelativeLayoutFeatureExtractor(conf)
            obj = obj@RelativeFeatureExtractor(conf)
        end
        function x = extractFeatures(obj,imageID,regions,pairs)
            %             rprops = struct;
            x = {};
            
            debug_ = false;
            
            %             d = cellfun(@(x) bwdist(x) <= 2, regions,'UniformOutput',false);
            rois = {};
            for iPair = 1:size(pairs,1)
                pair1 = pairs(iPair,1);
                pair2 = pairs(iPair,2);
                r1 = regions{pair1};
                r2 = regions{pair2};
                [y1,x1] = find(r1);
                xmin = min(x1);xmax = max(x1);
                ymin = min(y1); ymax = max(y1);
                [y2,x2] = find(r2);
                % find relative quadrant...
                y2 = mean(y2);
                x2 = mean(x2);
                if (x2 < xmin)
                    x_quad = 1;
                elseif (x2 > xmax)
                    x_quad = 3;
                else
                    x_quad = 2;
                end
                if (y2 < ymin)
                    y_quad = 1;
                elseif (y2 > ymax)
                    y_quad = 3;
                else
                    y_quad = 2;
                end
                
                xx = zeros(3,3);
                xx(y_quad,x_quad) = 1;
                
%                 clf; subplot(1,2,1); imagesc(r1+2*r2); axis image;
%                 hold on;
%                 bbox = [xmin ymin xmax ymax];
%                 plotBoxes2(bbox([2 1 4 3]),'g','LineWidth',2);
%                 subplot(1,2,2); imagesc(xx); axis image;
%                 pause;
                
                x{iPair} = xx(:);
            end
            % TODO - make this multiresolution / multi-angle, like you thought
            % previously.
        end
    end
end
