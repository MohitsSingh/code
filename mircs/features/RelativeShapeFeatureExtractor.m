classdef RelativeShapeFeatureExtractor < RelativeFeatureExtractor
    properties
    end
    
    methods
        function obj = RelativeShapeFeatureExtractor(conf)
            obj = obj@RelativeFeatureExtractor(conf)
        end
        function x = extractFeatures(obj,imageID,regions,pairs)
            %             rprops = struct;
            x = {};
            
            debug_ = false;
            logPolarMask = getLogPolarMask(10,10,4);
            
%             d = cellfun(@(x) bwdist(x) <= 2, regions,'UniformOutput',false);
            for iPair = 1:size(pairs)
                %                                 if (mod(iPair,100)==0) ,100*iPair/length(pairs) ,end
                %                 iPair
                
                pair1 = pairs(iPair,1);
                pair2 = pairs(iPair,2);
                r1 = regions{pair1};
                r2 = regions{pair2};
%                 d1 = d{pair1};
%                 d2 = d{pair2};
                                d1 = bwdist(r1) <= 2;
                                d2 = bwdist(r2) <= 2;
                
                % find the boundary, which is < 1 far from both groups
                boundaryRegion = d1 & d2;
                %                 boundaryRegion = (d1<=2 & d2<=2);
                if (nnz(boundaryRegion)==0)
                    x{iPair} = NaN;
                    continue;
                end
                z = r1 - r2; % z is the shape by merging the regions, with
                % sign indicating the region order
                % get the local shape.
                [ii,jj] = find(boundaryRegion);
                bbox = pts2Box([jj ii]);
                bbox = round(makeSquare(bbox));
                %                 mask_boundary = arrayCrop(boundaryRegion,bbox([2 1]),bbox([4 3]),0);
                %                 [s_bnd] = getLogPolarShape(mask_boundary);
                bbox = round(inflatebbox(bbox,2,'both',false));
                mask = arrayCrop(z,bbox([2 1]),bbox([4 3]),0);
                if (~debug_)
                    
                    mask1 = arrayCrop(r1,bbox([2 1]),bbox([4 3]),0);
                    mask2 = arrayCrop(r2,bbox([2 1]),bbox([4 3]),0);
                    s1 = getLogPolarShape(mask1,[],[],logPolarMask);
                    s2 = getLogPolarShape(mask2,[],[],logPolarMask);
                    %[s] = getLogPolarShape(mask,[],[],logPolarMask);
                    s = [s1;s2];
                else
                    [s,s_vis] = getLogPolarShape(mask,[],[],logPolarMask);
                    
                    clf;
                    imshow(r1+2*r2+4*boundaryRegion,[]);
                    hold on;
                    plotBoxes2(bbox([2 1 4 3]),'g');
                    imshow(s_vis);
                    pause;
                    % % %                 x{iPair} = [s_bnd;s];
                    
                end
                 if (obj.doPostProcess)
                    s = s/(sum(s.^2)^.5);
                 end
                x{iPair} = s;
            end
            x = cat(2,x{:});
            
        end
        % TODO - make this multiresolution / multi-angle, like you thought
        % previously.
    end
end
