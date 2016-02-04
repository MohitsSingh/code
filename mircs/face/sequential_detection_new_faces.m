echo off;
if (~exist('toStart','var'))
    initpath;
    config;
    
    addpath(genpath('/home/amirro/code/3rdparty/objectness-release-v2.0/'));
    params = defaultParams([pwd '/']);

    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    %faceModel = learnFaceModel_(conf);    
end

imageSet = imageData.test;
cur_t = imageSet.labels;
conf.get_full_image = true;
allScores = -inf(size(cur_t));
debug_ = true;
% initUnaryModels
%%
close all;

% 505, the Obama image
for k = 1:length(cur_t)
%     for k = 1:
%     482
    %  for k = 483    
%                 for k = [ 549   551 552 556 476 587 492 496 497 505 506 519 525 540 546 ] % 526
    %     for k = 476
    k
    imageInd = k;
%     if (~isinf(allScores(k)))
%         continue;
%     end
    currentID = imageSet.imageIDs{imageInd};
    if (~any(strfind(currentID,'drinking')))
%         continue
    end
    if(~cur_t(k))
%                 continue;
    end
    curTitle = '';    
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
    
    if (exist(ucmFile,'file'))
        load(ucmFile);
    end    
    
    
    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    ovp = boxesOverlap(faceBoxShifted,segmentBoxes)';
             
    [regions,regionOvp,G] = getRegions(conf,currentID,false);
    
%     
%     subplot(1,2,1); imagesc(I); axis image;
%     subplot(1,2,2); imagesc(ucm); axis image;
%     pause;
%     
    % find the single best overlapping segments
    
%      displayRegions(I,regions,ovp,-1,1);pause;
%      continue;
% s = 1;
% I1 = I;
% ddd = 100;
%     if (size(I,1) > ddd)
%         
%         s = ddd/size(I,1);        
%         I1 = imResample(I,[ddd size(I,2)*s],'bilinear');
%     end

%     local_segmentation(I1,faceBoxShifted*s);
    
%    
%     pause;
%     continue;
%     
    %[regionConfs] = applyModel(conf,currentID,partModels);
%     L_conf = load(fullfile('~/storage/res_s40',strrep(currentID,'.jpg','.mat')));
%     regionConfs = L_conf.regionConfs;
%     dpmResPath = fullfile(conf.dpmDir,strrep(currentID,'.jpg','.mat'));
%     load(dpmResPath);
    
    
    
    
    
    % %
    % %     T_ovp =.5; % don't show overlapping regions...
    % %     region_sel = suppresRegions(regionOvp, T_ovp); % help! we're being oppressed! :-)
    % %
    
    if (debug_)
    origBoundaries = ucm<=.1;
    segLabels = bwlabel(origBoundaries);
    segLabels = imdilate(segLabels,[1 1 0]);
    segLabels = imdilate(segLabels,[1 1 0]');
    
    S = medfilt2(segLabels); % close the small holes...
    segLabels(segLabels==0) = S(segLabels==0);
    
    %     conf.get_full_image = false;
    segImage = paintSeg(I,segLabels);
    clf;
    subplot(1,2,1); imagesc(I); axis image;
    hold on;
       plotBoxes2(faceBoxShifted([2 1 4 3]));
    plotBoxes2(lipRectShifted([2 1 4 3]));
    
    
    subplot(1,2,2); imagesc(ucm); axis image; pause; continue
    
    
%     subplot(1,2,2); imagesc(segImage); axis image;hold on;
    
% imshow(I)
% E = edge(rgb2gray(I),'canny');
% %  
    
    % is there a non-face object within the area of the face?
    
    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    
    ovp = boxesOverlap(lipRectShifted,segmentBoxes)';
    ovp_neg = boxesOverlap(faceBoxShifted,segmentBoxes)';
    areas = [props.Area];
    
%     [~,selBox] = imcrop(I);
%     selBox = imrect2rect(selBox);
    selBox = faceBoxShifted;
    [~,~,faceSize] =  BoxSize(selBox);
    
    s1 = col(ovp > 0);
    %     s2 = col( (areas./faceSize) >= .05 & (areas./faceSize) <= 2);
    s2 = col( (areas/faceSize) >= .05 & (areas/faceSize) <= 1);
    %     s2 = col( (areas./faceSize) >= 0 & (areas./faceSize) <= 2);
    s3 = segmentBoxes(:,2) >= faceBoxShifted(2);
    
    sel_ = s1 & s2 & s3;
    
    %boxes = runObjectness(I,1000);
    
    B = load(fullfile('~/storage/objectness_s40',strrep(currentID,'.jpg','.mat')));
    %
    [map,counts] = computeHeatMap(I,B.boxes(1:100,:),'sum');
    %    imagesc(map); axis image
    %
        Z = imfilter(map,fspecial('gauss',25,8));
   subplot(1,2,2);   imagesc(Z); axis image; 
% pause; continue;
%     
    z = cellfun(@(x) mean(Z(x)),regions);
    z = z/max(z(:));-1*ovp_neg';
%     displayRegions(I,regions(sel_),z(sel_),0,5);
displayRegions(I,regions(sel_),z(sel_),0,5);

    % compute the objectness of each rectangle: 
%     ovp_seg_obj = boxesOverlap(segmentBoxes,B.boxes);
%     weights = sum(ovp_seg_obj,2);
%     scores = ovp_seg_obj*B.boxes(:,5);
%     scores = scores./weights;
%     
%       displayRegions(I,regions,scores,0,10);
%         continue;
%       r = zeros(size(regions{1}));
%       for rr = 1:length(regions)
%           r = r+regions{rr}*scores(rr);
%       end
%       imagesc(r)
%       pause;
  
    % 
%     
%     ;pause; continue;
    %segAvg(boxes,regions);
    
    %[map,counts] = computeHeatMap(I,boxes,'max');
    %figure,imshow(map,[]);
    %figure,imshow(map./(counts+eps),[])
    
%     displayRegions(I,regions(17),-areas(17)./faceSize)
    
    
    % has to be:
    % large enough relative to face
    
    
    plotBoxes2(segmentBoxes(sel_,[2 1 4 3]),'g');
%     pause;
    continue;
    
    
    
    top_k = 50; % keep only the top 5 segments for each detection...
    %     regionSel_f = find(region_sel);
    T_ovp =1;
    end
    region_sel = suppresRegions(regionOvp, T_ovp); % help! we're being oppressed! :-)
    
% % %     subSel = false(size(region_sel));
% % %     allScores = zeros(length(regionConfs),length(regionSel_f));
% % %     
% % %     
% % %     for iModel = 1:length(regionConfs)
% % %         curScores = regionConfs(iModel).score(region_sel);
% % %         curScores(isnan(curScores)) = -inf;
% % %         allScores(iModel,:) = curScores;
% % %     end
% % %     [r,ir] = sort(allScores,2,'descend');
% % %     
% % %     tops = unique(row(ir(:,1:top_k))); % NOTE: this may be actually less than top_k*length(models)
% % %     region_sel = false(size(region_sel));
% % %     region_sel(regionSel_f(tops)) = true;
% % %     
    
    for iModel = 1:length(regionConfs)
        regionConfs(iModel).score = regionConfs(iModel).score(region_sel);
        nans = isnan(regionConfs(iModel).score);
        regionConfs(iModel).score(nans) = -inf;
    end
    
    regions = regions(region_sel);
    regionOvp = regionOvp(region_sel,region_sel);
    
    %     selBox = faceBoxShifted;
    selBox = lipRectShifted;
%     try
%         L = load(fullfile('~/storage/relativeFeats_s40',strrep(currentID,'.jpg','.mat')));
%     catch e
%         continue
%     end
    %     relativeShapes = relativeShapes(:,region_sel);
    
%     scores = L.scores;
    
    
    
%     for iScore = 1:length(scores)
%         if (isempty(scores{iScore}))
%             scores{iScore} = zeros(size(L.pairs,1),1);
%         end
%     end
%     scores = cat(2,scores{:})';
%     scores_ = region2EdgeSubset(G,scores,region_sel);
    
    
    
    G = G(region_sel,region_sel);
    
    
    faceBox = faceBoxShifted;
    
    conf.demo_mode = false;
    faceConfs = regionConfs(1).score;
    %       faceConfs = regionConfs(2).score; % cup
    [ff,iff] = sort(faceConfs,'descend');
    
    %      displayRegions(I,regions(iff(1:3)),faceConfs(iff(1:3)));
    %      continue
    
    [ovp_face,ints_face,areas_face] = boxRegionOverlap(faceBox,regions,size(regions{1}));
    
    % find an internal region with maximal area...
%     ints_face./areas_face
    
    %displayRegions(I,regions,ovp_face,0,5);
    [r,ir] = max(ovp_face);
    faceRegion = ir;
    % find neighbors with high cup-score, this will be the score...
    faceNeighbors = find(G(ir,:));
    if (isempty(faceNeighbors))
        continue;
    end
    faceScores = regionConfs(1).score(faceNeighbors);
    [q,iq] = max(faceScores);
%     displayRegions(I,{.35*regions{faceRegion}+.75*regions{faceNeighbors(iq)}});


    clf,subplot(1,2,1);imagesc(.35*regions{faceRegion}+.75*regions{faceNeighbors(iq)}); axis image
    hold on; plotBoxes2(faceBox([2 1 4 3]),'r','LineWidth',3);
    subplot(1,2,2); imagesc(I);axis image; title(num2str(q));
    pause; 
    continue
%     displayRegions(I,{regions{faceNeighbors(iq)}});
% %     displayRegions(I,regions(ir));
    
%     [parts,allRegions,scores] = followSegments3(conf,regions,G,regionConfs,I,selBox,faceBox,regionOvp,[],[]);
    
    allScores(k) = allRegions{1}(2);
    %         [],[]);%relativeShape,relativeShapes_);
    %     Z = zeros(dsize(I));
    %     for pp = 1:length(parts{1})
    %         Z(regions{allRegions{1}(pp)reg = pp;
    %     end
    
    %     pause;
    %
end
%%
