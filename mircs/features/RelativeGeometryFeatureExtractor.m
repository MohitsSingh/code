classdef RelativeGeometryFeatureExtractor < RelativeFeatureExtractor
    properties
    end
    
    methods
        function obj = RelativeGeometryFeatureExtractor(conf)
            obj = obj@RelativeFeatureExtractor(conf)
        end
        function x = extractFeatures(obj,imageID,regions,pairs)
            %             rprops = struct;
            x = {};
            
            debug_ = true;
            regionGeometry =  obj.getGeometry(regions);
            d = cellfun(@(x) bwdist(x) <= 2, regions,'UniformOutput',false);
            for iPair = 1:size(pairs)
                %                                 if (mod(iPair,100)==0) ,100*iPair/length(pairs) ,end
                %                 iPair
                
                pair1 = pairs(iPair,1);
                pair2 = pairs(iPair,2);
                r1 = regions{pair1};
                r2 = regions{pair2};
                d1 = d{pair1};
                d2 = d{pair2};
               
                % find the boundary, which is < 2 far from both groups
                boundaryRegion = d1 & d2;               
                if (nnz(boundaryRegion)==0)
                    x{iPair} = NaN;
                    continue;
                end
                
                % sign indicating the region order
                % get the local shape.
                [ii,jj] = find(boundaryRegion);
                
                
                areaRatio = regionGeometry(pair1).Area/regionGeometry(pair1).Area;
                area1 =  regionGeometry(pair1).Area;
                area2 =  regionGeometry(pair2).Area;
%                 
%             
                x{iPair} = [areaRatio;area1;area2];
            end
            x = cat(2,x{:});
            
        end
        % TODO - make this multiresolution / multi-angle, like you thought
        % previously.        
    end
    methods (Hidden = true)
        function x = getGeometry(obj,regions)
           regions = fillRegionGaps(regions);
           x = cellfun(@(x) regionprops(x,'Area'),regions,'UniformOutput',false);
           x = cat(1,x{:});
        end
    end
end
