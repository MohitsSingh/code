function gt_configurations = sample_configurations(imgData,I,min_gt_ovp,gt_graph,params,prevTheta)
gt_configurations = {};
cur_config = struct('bbox',gt_graph{1}.bbox,'mask',gt_graph{1}.roiMask);
cur_config.endPoint = boxCenters(cur_config.bbox);
cur_config.xy = box2Pts(cur_config.bbox);
if nargin < 6    
    cur_config.theta = 0;
else
    cur_config.theta = prevTheta
end
faceBox = imgData.faceBox;
scaleFactor = faceBox(4)-faceBox(2);
if (min_gt_ovp > 0)
    for iNode = 2:length(gt_graph) % first node is given
        prev_endpoint = cur_config(iNode-1).endPoint;
        % find overalpping regions...
        curMask = gt_graph{iNode}.roiMask;
        if strcmp(params.cand_mode,'polygons')
            rois = getCandidateRoisPolygons(prev_endpoint,scaleFactor,params.sampling,true);
            poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);
            [ovp,ints,uns] = regionsOverlap(poly_masks,curMask);
            %displayRegions(I,poly_masks,ovp)
            [r,ir] = sort(ovp,'descend');
            ir = ir(1:3);
            thetaToKeep = ir;
            %[ovp,ints,uns] = regionsOverlap(poly_masks,curMask);
            rois = getCandidateRoisPolygons(prev_endpoint,scaleFactor,params.sampling,false,thetaToKeep);
            % %         curRoi = findBestPoly(rois,curMask);
            poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);
            %         displayRegions(I,poly_masks);
            [ovp,ints,uns] = regionsOverlap(poly_masks,curMask);
            [r,ir] = sort(ovp,'descend');
            bestRoi = rois{ir(1)};
            cur_config(iNode).endPoint = bestRoi.endPoint;
            cur_config(iNode).mask = poly_masks{ir(1)};
            cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
            cur_config(iNode).theta = bestRoi.theta;
            cur_config(iNode).xy = bestRoi.xy;
            %         displa  displayRegions(I,poly_masks);yRegions(I,poly_masks,ovp)
            %         [x3, y3] = polybool(operation, x1, y1, x2, y2, varargin)
        elseif strcmp(params.cand_mode,'boxes')
            
            
            beAggressiveAboutGT = true;
            if beAggressiveAboutGT
                maskBox = makeSquare(region2Box(curMask));
                boxedMasked = box2Region(maskBox,size2(I));
                xy = box2Pts(maskBox);
                cur_config(iNode) = struct('bbox',maskBox,'mask',boxedMasked,...
                    'endPoint',[],'xy',xy,'theta',0);
            else
                %curMask =  box2Region(makeSquare(region2Box(curMask),true),size2(I));
                %         displayRegions(I,poly_masks);
                rois = getCandidateRoisBoxes(cur_config(iNode-1).bbox,scaleFactor,params.sampling,I);
                poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);
                [ovp,ints,uns] = regionsOverlap(poly_masks,curMask);
                [r,ir] = sort(ovp,'descend');
                bestRoi = rois{ir(1)};
                cur_config(iNode).endPoint = bestRoi.endPoint;
                cur_config(iNode).mask = poly_masks{ir(1)};
                cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
                cur_config(iNode).theta = bestRoi.theta;
                cur_config(iNode).xy = bestRoi.xy;
                
            end            
        else % segments.
            curMask = gt_graph{iNode}.roiMask;
            [yy,xx] = find(curMask);
            dists_to_prev = l2([xx yy],prev_endpoint);
            [r,ir] = max(dists_to_prev);
            cur_config(iNode).endPoint = [xx(ir) yy(ir)];
            cur_config(iNode).mask = curMask;
            cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
            cur_config(iNode).theta = 0; % irrelevant in this case? 
            cur_config(iNode).xy = fliplr(bwtraceboundary2(curMask));
        end
        %rois = hingedSample(startPt,avgWidth,avgLength,thetas);                
    end
    
    gt_configurations = {cur_config};
    
else % sample random configurations
    nSamples = params.nSamples;
    for n = 1:nSamples
        isBad = false;
        for iNode = 2:length(gt_graph) % first node is given
            prev_endpoint = cur_config(iNode-1).endPoint;
            % find overalpping regions...
            if strcmp(params.cand_mode,'polygons')
                
                p = params.sampling;
                p.thetas = vl_colsubset(p.thetas,1,'random');
                p.widths = vl_colsubset(p.widths,1,'random');
                p.lengths = vl_colsubset(p.lengths(p.lengths>=p.widths),1,'random');
                
                rois = getCandidateRoisPolygons(prev_endpoint,scaleFactor,p,false);
                if (isempty(rois))
                    disp('11');
                end
                
                if ~all(inImageBounds([1 1 fliplr(size2(I))],rois{1}.xy))
                    isBad = true;
                    break;
                end
                poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);
                
                rois = rois{1};
                
                cur_config(iNode).endPoint = rois.endPoint;
                cur_config(iNode).mask = poly_masks{1};
                cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
                cur_config(iNode).theta = rois.theta;
                cur_config(iNode).xy = rois.xy;
                %         displayRegions(I,poly_masks,ovp)
                %         [x3, y3] = polybool(operation, x1, y1, x2, y2, varargin)
            elseif strcmp(params.cand_mode,'boxes')
                rois = getCandidateRoisBoxes(cur_config(iNode-1).bbox,scaleFactor,params.sampling,I);
                rois = vl_colsubset(row(rois),1,'Random');
                poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);                                
                bestRoi = rois{1};
                cur_config(iNode).endPoint = bestRoi.endPoint;
                cur_config(iNode).mask = poly_masks{1};
                cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
                cur_config(iNode).theta = bestRoi.theta;
                cur_config(iNode).xy = bestRoi.xy;
            else
                
                %[I_sub,mouthBox,candidates] = getCandidateRegions(params.conf,imgData);
                
                load(j2m('~/storage/fra_db_seg',imgData));candidates = cadidates; clear cadidates;
                masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
                masks = row(squeeze(mat2cell2(masks,[1,1,size(masks,3)])));
                % THE BOXES ARE FLIPPED!!!
                for t = 1:length(masks)
                    %clf; imagesc2(sc(cat(3,I_sub,im2double(candidates.masks{t})),'prob'));
                    clf; displayRegions(I,masks{t});
                    plotBoxes(candidates.bboxes(t,[2 1 4 3]));
                    dpc
                end
                
                % find 
                
% %                 for t = 1:length(candidates.masks)
% %                     %clf; imagesc2(sc(cat(3,I_sub,im2double(candidates.masks{t})),'prob'));
% %                     clf; displayRegions(I_sub,candidates.masks{t});
% %                     plotBoxes(candidates.bboxes(t,:));
% %                     dpc
% %                 end
                
                curMask = gt_graph{iNode}.roiMask;
                [yy,xx] = find(curMask);
                dists_to_prev = l2([xx yy],prev_endpoint);
                [r,ir] = max(dists_to_prev);
                cur_config(iNode).endPoint = [xx(ir) yy(ir)];
                cur_config(iNode).mask = curMask;
                cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
                cur_config(iNode).theta = 0; % irrelevant in this case?
                cur_config(iNode).xy = fliplr(bwtraceboundary2(curMask));
            end
            %rois = hingedSample(startPt,avgWidth,avgLength,thetas);
        end
        if ~isBad
            gt_configurations{end+1} = cur_config;
        end
    end
end



%             all_rois{end+1} = hingedSample(startPt,curWidth,curLength,thetas);
%         end
%         end



function r = findBestPoly(rois,curMask)
all_polygons = cellfun2(@(x) x.xy, rois);
r = 0;
%     roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
% roiPatches = cellfun2(@(x) rectifyWindow(I,round(x.xy),[avgLength avgWidth]),rois);


