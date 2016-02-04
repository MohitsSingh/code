function [configs,scores] = findBestConfiguration3(imgData,I,gt_graph,params,featureExtractor,...
    models_parts,models_links,startTheta,cands,sal_image)



gt_configurations = {};
cur_config = struct('bbox',gt_graph{1}.bbox);%,'mask',gt_graph{1}.roiMask);
cur_config.endPoint = boxCenters(cur_config.bbox);
cur_config.startPoint = cur_config.endPoint;
cur_config.xy = box2Pts(cur_config.bbox);
cur_config.theta = startTheta;
cur_config.score = 0; % no need to score the first one as we're looking for the argmax
faceBox = imgData.faceBox;
scaleFactor = faceBox(4)-faceBox(2);
nSamples = params.nSamples;
restrictAngle = true;
scores = cur_config.score;
configs = {cur_config};
expansionFactor = 5;

% get masks, boxes only in vicinity of face...
face_in_boxes = BoxIntersection2(cands.bboxes, imgData.faceBox);
[~,~,faceArea] = BoxSize(imgData.faceBox);
sel_ = face_in_boxes > 0 & face_in_boxes/faceArea < .7;
%showSortedBoxes(I,cands.bboxes(sel_,:),-face_in_boxes(sel_))
regionBoxes = cands.bboxes(sel_,:);
regions = cands.masks(sel_);

for iPart = 2%:length(gt_graph)
    [r,ir] = sort(scores,'descend');
    n = length(configs);
    ir = ir(1:min(n,expansionFactor));
    prev_configs = configs(ir);
    configs = {};
    scores = {};
    %
    for iConfig = 1:length(prev_configs) % iterate over candidates of depth -1
        cur_config = prev_configs{iConfig}; % get configuration to expand
        cur_score = cur_config(end).score;
        % find overlapping regions...
        % generate new patches around current
        
        %sample_restricted                              
        curRois = getCandidateRoisPolygons2( cur_config(end) ,...
            scaleFactor,params.sampling,restrictAngle);
                        
        conf.get_full_image = true;        
        localSaliencyImage = getLocalSaliency(imgData,2*scaleFactor,I);
        roiMasks = arrayfun2(@(x) poly2mask2(x.xy,size2(I)), curRois);
        roiAreas = cellfun3(@nnz,roiMasks);
        roiSaliency = cellfun3(@(x) sum(localSaliencyImage(x)),roiMasks);
        roiAreas = cellfun3(@nnz,regions);
        roiSaliency = cellfun3(@(x) sum(localSaliencyImage(x)),regions);
%         displayRegions(I,regions,roiSaliency./roiAreas)
%         [ovp1,int1] = regionsOverlap3(regions,roiMasks);
        BB = round(inflatebbox(boxCenters(cur_config(end).bbox),scaleFactor,'both',true));
        boxes2 = getRegionBoxes(regions2);
        new_bb = BoxIntersection(BoxUnion(boxes1),BoxUnion(boxes2));
        regions1 = cellfun2(@(x) cropper(x,new_bb),regions1);
        regions2 = cellfun2(@(x) cropper(x,new_bb),regions2);
        boxes1 = BoxIntersection(boxes1,new_bb);
        boxes2 = BoxIntersection(boxes2,new_bb);

% % %         [ovp1,int1] = regionsOverlap(regions,roiMasks,false);
        
% % %         [mm,imm] = max(ovp1,[],2);
        
        %%%%%%%%%%%%%%%%%%%%% saliency
       
        % try matching this with a superpixel map -- > for each candidate
        % region, find the best matching superpixel                                
        % match to each superpixel a rectangular region, only if it
        % "touches" or is close enough to the "endpoint"
        
        
        % find only regions in extended mouth area
% % %         BB_touching = cur_config(end).bbox;
% % %         
% % %         
% % %         BB = round(inflatebbox(boxCenters(cur_config(end).bbox),scaleFactor,'both',true));
% % %         BB_touching = cur_config(end).bbox;        
% % %         D_image = bwdist(box2Region(BB_touching,size2(I)));
% % %         z_clear = box2Region(BB,size2(I));
% % %         masks = cands.masks;
% % %         
% % %         % remove masks not in region of interets
% % %         
% % %         
% % %         SP = cands.superpixels;
% % %         SP(~z_clear) = 0;
% % %         u = unique(SP);
% % %         u(u==0) = [];
% % %         SP_new = zeros(size(SP));
% % %         tt = 0;
% % %         for t = 1:length(u)
% % %             z = SP==u(t);           
% % %             if nnz(z) <= 15
% % %                 continue
% % %             end
% % %             tt = tt+1;
% % %             SP_new(z) = tt;
% % %         end
% % %         SP = SP_new;
% % %         
% % %         rProps = regionprops(SP_new,D_image,'PixelIdxList','ConvexHull','Orientation','MajorAxisLength',...
% % %             'MinorAxisLength','MaxIntensity','MinIntensity');
% % %                 
% % %         minIntensities = [rProps.MinIntensity];
% % %         % maximal distance of 5 pixl
% % %         rProps(minIntensities >= 5) = [];
% % %         newMasks = {};
% % %         for t = 1:length(rProps)          
% % %             z = zeros(size2(I));
% % %             z(rProps(t).PixelIdxList) = 1;
% % %             newMasks{end+1} = z;
% % %         end
% % %         
% % %         for t = 1:length(rProps)
% % %             disp(length(rProps(t).PixelIdxList))
% % %             clf; imagesc(SP_new==t)
% % %             dpc
% % %         end
        
        
        
        
%         [m,im] = max(ovps,[],2);        
%         [r,ir] = sort(m,'descend');
%         for t = 1:length(m)
%             clf; imagesc2(I);            
%             plotPolygons(polygons{ir(t)});
%             title(sprintf('%f (%d)',r(t),t));
%             dpc
%         end
% 
%         
        
        %
        %         for m = 1:length(curRois)
        %             clf; imagesc2(I)
        %             plotPolygons({curRois(m).xy})
        %             dpc(.1)
        %         end
        
        %        nextRois = sampleAround(cur_config(end),nSamples,scaleFactor,params,I,restrictAngle);
        %        nextRoiBoxes = cellfun3(@(x) x.bbox,nextRois);
        % find large overlap between candidates and current boxes
        % % %         [ovp,int] = boxesOverlap(cands.bboxes,nextRoiBoxes);
        % % %         [m,im] = max(ovp,[],2);
        % % %         [n,in] = max(int,[],2);
        % % %         sel_ = m > .35;
        % % %         %displayRegions(I,cands.masks,m);
        % % %         my_masks = cands.masks(sel_);
        % % %         my_boxes = cands.bboxes(sel_,:);
        % % %         prevCenter = round(boxCenters(cur_config(end).bbox));
        % % %         prevMask = box2Region([prevCenter prevCenter+1],size2(I));
        % % %         Z = bwdist(prevMask);
        % % %         min_dists = cellfun3(@(x) min(Z(x)),my_masks);
        % % %         sel_ = min_dists <= scaleFactor*.2;
        % % %         my_boxes = my_boxes(sel_,:);
        % % %         my_masks = my_masks(sel_);
        % % %         displayRegions(I,my_masks)
        % % % %         nextRois = cellfun2(@(x) {x}, nextRois);
        % % % %         curScores = zeros(size(nextRois));
        [partFeats,linkFeats,patches] = getPartFeats(I,cur_config,curRois,featureExtractor,params);
        curScores = models_parts(iPart).w'*partFeats;
        if (params.interaction_features && ~isempty(linkFeats))
            linkScores = models_links(iPart-1).w'*linkFeats;
            curScores = curScores+linkScores;
        end
        
        %curPartFeatures = configurationToFeats2(I,nextRois,featureExtractor,params);
        
        %curScores = w_parts{iPart}'*cat(2,curPartFeatures{:});
        % keep scores & configurations
        for iCandidate = 1:length(curRois)
            R = curRois{iCandidate};
            R.score = curScores(iCandidate)+cur_score;
            configs{end+1} = [cur_config,R];
            scores{end+1} = R.score;
        end
    end
    scores = [scores{:}];
    
end


function [localSaliencyImage,F] = getLocalSaliency(imgData,scaleFactor,I)
zFactor = .7;
F = round(inflatebbox(imgData.mouth,scaleFactor*zFactor,'both',true));
I_for_sal = im2uint8(cropper(I,F));
maxImageSize = 50;
I_for_sal = imResample(I_for_sal,[maxImageSize maxImageSize]);
sal_opts.maxImageSize = maxImageSize;
sal_opts.show = false;
Z = [];
%sizeRatio = [10 15 20]
sizeRatio = 10
for curSizeRatio = sizeRatio
    spSize = (maxImageSize/curSizeRatio)^2;
    %     spSize = 50;
    %     I = imResample(I,size2(sal1),'bilinear');
    sal_opts.pixNumInSP = spSize;
    sal_opts.useSP = true;
    [sal1,sal_bd,resizeRatio,sp_data] = extractSaliencyMap(I_for_sal,sal_opts);
    if isempty(Z)
        Z = sal1;
    else
        Z = Z+sal1/length(sizeRatio);
    end
end

localSaliencyImage = transformBoxToImage(I, Z, F, false);




function [ovps,ints] = polysRegionsOverlap(polys,superpixels)
polyBoxes = cellfun3(@(x) pts2Box(x),polys);
B = BoxUnion(polyBoxes);
B(1:2) = floor(B(1:2));
B(3:4) = ceil(B(3:4));
superpixels = cropper(superpixels,B);
superpixels = RemapLabels(superpixels);
masks ={};
for t = 1:length(unique(superpixels(:)))
    masks{t} = superpixels==t;
end
sz = size2(superpixels);
poly_masks = cellfun2(@(x) poly2mask2(bsxfun(@minus,x,B(1:2)),sz),polys);
[ovps,ints] = regionsOverlap(poly_masks,masks);

% sort by maximal overlap and show for each polygons it's overlapping
% region.

% end

% visualizeConfigurations(I,configs,[scores{:}],5,.5);

% if ~isBad
%     gt_configurations{end+1} = cur_config;
% end




% % %     function rois = getCandidateRoisSliding(startPt,scaleFactor,samplingParams,onlyTheta,thetaToKeep)
% % %
% % %         if nargin < 5
% % %             thetaToKeep = true(size(samplingParams.thetas));
% % %         end
% % %         thetas = samplingParams.thetas(thetaToKeep);
% % %         lengths = samplingParams.lengths*scaleFactor;
% % %         widths = samplingParams.widths*scaleFactor;
% % %         if onlyTheta
% % %             lengths = max(lengths);
% % %             widths = mean(widths);
% % %         end
% % %         all_rois = {};
% % %
% % %         for iLength = 1:length(lengths)
% % %             for iWidth = 1:length(widths)
% % %                 curLength = lengths(iLength);
% % %                 curWidth = widths(iWidth);
% % %                 if curWidth > curLength
% % %                     continue
% % %                 end
% % %                 all_rois{end+1} = hingedSample(startPt,curWidth,curLength,thetas);
% % %             end
% % %         end
% % %
% % %         rois = cat(2,all_rois{:});
% % %         %     roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
% % %         % roiPatches = cellfun2(@(x) rectifyWindow(I,round(x.xy),[avgLength avgWidth]),rois);
