echo off;
if (~exist('toStart','var'))
    initpath;
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    imageSet = imageData.train;
    cur_t = imageSet.labels;
    allScores = -inf*ones(size(cur_t));
    frs = {};
    pss = {};
    hand_scores = {};
    f = find(cur_t);
    strawInds_ = f([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54]); % for train only!!
    strawInds = strawInds_;
    Zs = {};
    fhog1 = @(x) fhog(im2single(x),4,9,.2,0);    
    % get the ground truth for cups...
        [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    gtParts = {groundTruth.name};
    isObj = cellfun(@any,strfind(gtParts,'cup'));    
    % when is the cup near the mouth enough? 
    groundTruth = groundTruth(isObj);               
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t)
end


% should improve the face alignment...

%%
 for k = 1:length(groundTruth)
        currentID = groundTruth(k).sourceImage;        
        ff = find(cell2mat(cellfun2(@any,strfind(imageSet.imageIDs,currentID))));
        if (isempty(ff))
            continue;
        end
        I = getImage(conf,currentID);
        poly = groundTruth(k).polygon;
        bw = poly2mask(poly.x,poly.y,size(I,1),size(I,2));
        clf;
        imagesc(I);axis image        
        lipRectShifted = imageSet.lipBoxes(ff,:);
        hold on; plot(poly.x,poly.y,'r-','LineWidth',2);
        hold on; plotBoxes2(lipRectShifted([2 1 4 3]),'g');
        pause;
    end
    
%%
close all;
debug_ = true;

% ff = find(cur_t);
% 505, the Obama image
% for k = ff([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54])'
% for q = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ] % 526
% iv = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ];
% iv = 1:length(cur_t)
% iv = strawInds_;
for q = 1:length(cur_t)
% for q = 390
    
%     for q = 380:1000
        
    k = iv(q);
    if (~debug_ && ~isempty(allFeats{k}))
        continue;
    end
    imageInd = k;
    
    if (~cur_t(k))
        continue;
    end
    q
    currentID = imageSet.imageIDs{imageInd};
        
    curTitle = '';
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    lipRectShifted = imageSet.lipBoxes(imageInd,:);
    facePts = imageSet.faceLandmarks(imageInd).xy;
    facePts = boxCenters(facePts);
    if (debug_)
        clf;
        subplot(2,3,1);
        imagesc(I); axis image; hold on;
        plotBoxes2(faceBoxShifted([2 1 4 3]));
        plotBoxes2(lipRectShifted([2 1 4 3]),'m');
        plot(facePts(:,1),facePts(:,2),'r.');
    end
    box_c = round(boxCenters(lipRectShifted));
    
    sz = faceBoxShifted(3:4)-faceBoxShifted(1:2);
    %bbox = round(inflatebbox([box_c box_c],floor(sz/1.5),'both',true));
    bbox = round(inflatebbox([box_c box_c],floor(sz/2),'both',true));
           
    if (debug_)
        plotBoxes2(bbox([2 1 4 3]),'g');
    end
    
    if (any(~inImageBounds(size(I),box2Pts(bbox))))
        if (~debug_)
            allScores(k) = -10^6;
        end
        continue;
    end

    I_sub_color = I(bbox(2):bbox(4),bbox(1):bbox(3),:);   
    I_sub = rgb2gray(I_sub_color);
    
    II = imcrop(I)
    II = cropper(I,inflatebbox(faceBoxShifted,[1.1 1.1],'both',false));
    imshow(II)
    detect_landmarks(conf,{imresize(II,3)},1,false);
    
    if (debug_)
        subplot(2,3,2);
        imagesc(I_sub_color); axis image; colormap gray;
    end
    
    [M,O] = gradientMag(im2single(I_sub));
    %     imagesc(M); axis image; colormap gray;
    %     imagesc(O); axis image; colormap gray;
    
    ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
    load(ucmFile); % ucm
    
    %     E = edge(I_sub,'canny');
        
    subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
    E = subUCM;
              
    xy = imageSet.faceLandmarks(k).xy;
    xy_c = boxCenters(xy);
%     clf;
%     imagesc(I); 
%     hold on; 
     
    chull = convhull(xy_c);
    
    % find the occluder!! :-)
    c_poly = xy_c(chull,:);
    
%     [result, resultNG] = PCA_Saliency(I_sub_color);
%     local_segmentation(I,poly2mask(c_poly(:,1),c_poly(:,2),size(I,1),size(I,2)));
    
    c_poly = bsxfun(@minus,c_poly,bbox(1:2));
    
    
% %     subplot(2,3,4); hold on;plot(c_poly(:,1),c_poly(:,2),'g-');
% % %     figure,imagesc(subUCM)
% %     regions_sub = combine_regions_new(subUCM,.1);                
% % %     regionOvp_ = regionsOverlap(regions_);
% % %     G_ = regionAdjacencyGraph(regions_);
% % %     displayRegions(I_sub_color,regions_);
% %     
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));
       
    bb = bwperim(face_mask);
    bd = bwdist(bb);
    
%     
%     regions = getRegions(conf,currentID);
%     regionMap = zeros(size(ucm));
%     for k = 1:length(regions)
%         curRegion = regions{k};
%         if ~any(regionMap(curRegion))
%             regionMap(curRegion) = k;
%         else
%             break;
%         end
%     end
%     
    
%     pause;continue;
%     figure,imagesc(bd)
    
    %figure,imagesc(exp(-bd.^2/10)).*subUCM)
%     figure,imagesc(subUCM)
%     paintLines(zeros(size(E)),)
    
    % decide which edges are close enough to the face border and remove
    % them.
%     E = E.*bd>=3;
    
    %E = E.*face_mask;
    
    if (debug_)
        subplot(2,3,4);imagesc(E);
    end
    
% %             
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));
    regions_sub = combine_regions_new(subUCM,.2);
    regions_sub = fillRegionGaps(regions_sub);    
%     displayRegions(I_sub_color,regions_sub)
    perims = cellfun2(@bwperim,regions_sub);
    boundaries = cellfun2(@bwboundaries,regions_sub);
    % take only one of each.
    boundaries = cellfun2(@(x) x{1},boundaries);
    lines = lineseg(boundaries,2);
    candidates = splitByDirection(lines);
%     imagesc(E); hold on;
%     drawedgelist(lines,size(E),2,'rand');
    
    allSegs = {};
%     pause; continue;
%     for k = 1:length(perims)
%         p = perims{k};
%         bwtraceboundary
%     end
%     
%     allSegs = cellfun2(@(x) edgelink(x, []),perims);
%             
%     [ovp_sub,ints_sub,areas_sub] = boxRegionOverlap(face_mask,regions_sub);
%     
%     % features to consider: area of intersection
%     ucmStrengths = cellfun(@(x) mean(subUCM(imdilate(x,ones(3)) & subUCM > 0)),regions_sub);
% %     displayRegionimages(subUCM,regions_sub,ucmStrengths);
%     
%     %regionFeats = [ints_sub;areas_sub;ones(size(areas_sub))*nnz(face_mask);ucmStrengths]';
%     
% %     regionFeats = [ints_sub./areas_sub;1-ints_sub./areas_sub;ints_sub/nnz(face_mask);ucmStrengths]';
%     
%     curScore = areas_sub < .5*nnz(face_mask) &...
%         ints_sub./areas_sub > .3 &...
%         ints_sub./areas_sub < .8;
    
%     %curScore = regionFeats*[1 1 -1 1]';%+regionFeats(:,1) > .3 & regionFeats(:,2) < .7;
%     [s,is] = sort(curScore,'descend');
%     subplot(2,3,3); montage2(cat(3,regions_sub{is}));
%     pause;continue;
    
    E = E>.2;
    if (~any(E(:)))
        continue;
    end
    E = bwmorph(E,'thin',Inf);
            
    
%     E = imresize(double(E),.25,'bilinear');
    if (nnz(E) < 3)
        continue;
    end
%     [seglist,edgelist] = processEdges(E);
  
%     if (debug_)
%         drawedgelist(edgelist,size(E),2,'rand');
%     end
%     
%     if (debug_)
% %         clc;
%     end
    
    %   pause;continue;
%     candidates = findConvexArcs2(seglist,E,edgelist);
    %     candidates = candidates(6)
    
    % make sure left point is first.
    candidates = fixSegLists(candidates);
    candidates = seglist2edgelist(candidates);
    lengths = cellfun(@length,candidates);
    candidates = candidates(lengths > 2);
    
    if (isempty(candidates))
        continue;
    end
    % make sure that left-right distance is large enough (in terms of
    % search window)
    
    % extract some features from these candidates
    pFirst = cell2mat(cellfun(@(x) x(1,:),candidates,'UniformOutput',false)');
    pLast =  cell2mat(cellfun(@(x) x(end,:),candidates,'UniformOutput',false)');
    pMean = cell2mat(cellfun(@(x) mean(x),candidates,'UniformOutput',false)');
        
    horzDist = (pLast(:,2)-pFirst(:,2))./size(I_sub,2);
    
    isConvex = false(size(candidates));
    u = pLast-pFirst;
    
    % make sure that the base of the contour isn't too vertical    
    
    for ic = 1:length(candidates)
        v = bsxfun(@minus,candidates{ic}(2:end-1,:),pFirst(ic,:));
        curCross = u(ic,2)*v(:,1)-u(ic,1)*v(:,2);
        isConvex(ic) = ~any(curCross>0);
    end
    
    if (~any(isConvex))
        continue;
    end
    
    horzDist = horzDist(isConvex);
    pMean = pMean(isConvex,:);
    
    candidates = candidates(isConvex);
    verticality = abs(u(isConvex,1)./u(isConvex,2));
    %     candidates = candidates(verticality<2);
    if (isempty(candidates))
        continue;
    end
    %     pMean = pMean(verticality<2);
    contourLengths = cellfun(@(x) sum(sum(diff(x).^2,2).^.5),candidates);        
    candidateImages = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),ones(3)),...
        candidates);
    
    ims = cellfun2(@(x) I_sub+ imdilate(paintLines(false(size(E)),seglist2segs({x})),ones(3)),...
        candidates);
    ucmStrengths = cellfun(@(x) mean(subUCM(x & subUCM > 0)),candidateImages);
        
    skinprob = computeSkinProbability(double(im2uint8(I_sub_color)));
    normaliseskinprob = normalise(skinprob);
                   
    strel_ = [ones(1,5),zeros(1,3),zeros(1,5)]';    
    ims_up = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),strel_),...
        candidates);
    ims_down = cellfun2(@(x) imdilate(paintLines(false(size(E)),seglist2segs({x})),flipud(strel_)),...
        candidates);    
    ims_ = cat(3,ims_up{:})-cat(3,ims_down{:});
    
    % normalize filters to have norm 1
    ims_ = bsxfun(@rdivide,ims_,sum(sum(abs(ims_))));
    
    
    
    
    % find the interesection points (if any) of the candidates with the
    % given face outline. 
    
    c_poly = xy_c(chull,:);
    c_poly = bsxfun(@minus,c_poly,bbox(1:2));
    
    % turn poly into a sequence of line segments
    face_segs = [c_poly(1:end-1,:),c_poly(2:end,:)];
    
    
    nIntersections = zeros(size(candidates));
    
      for iCandidate = 1:length(candidates)
        out = lineSegmentIntersect(seglist2segs(candidates(iCandidate)),face_segs(:,[2 1 4 3]));
        nIntersections(iCandidate) = nnz(out.intAdjacencyMatrix);
      end
    
    
%     for iCandidate = 1:length(candidates)
%         out = lineSegmentIntersect(seglist2segs(candidates(iCandidate)),face_segs(:,[2 1 4 3]));
%         nIntersections(iCandidate) = nnz(out.intAdjacencyMatrix);
%         imagesc(E); axis image; hold on;
%         drawedgelist({c_poly(:,[2 1])},size(E),2,'g');
%         drawedgelist(candidates(iCandidate),size(E),2,'m');
%         if (nIntersections(iCandidate)==0)
%             continue;
%         end
%         disp(['number of intersections: ' num2str(nIntersections(iCandidate))]);
%         
%         [~,~,x] = find(out.intMatrixX);
%         [~,~,y] = find(out.intMatrixY);
%         nans = isnan(x) | isnan(y);
%         x = x(~nans);
%         y = y(~nans);
%         plot(y,x,'rs','MarkerFaceColor','r');
%         plot(c_poly(:,1),c_poly(:,2),'g-');
%         pause(.01)
%         
%     end
%     
    
%     hold on; plot(c_poly(:,1),c_poly(:,2),'r-');
    
%     % snap the polygon to the nearest 
    
    
    bw = poly2mask(xy_c(chull,1),xy_c(chull,2),size(I,1),size(I,2));
    bw = imerode(bw(bbox(2):bbox(4),bbox(1):bbox(3)),ones(3));
    insides = cellfun(@(x) nnz(bw.*paintLines(false(size(E)),seglist2segs({x}))>0),candidates);
    insides = insides/size(E,1);
    
%     myfun= @(x) nnz(bw.*paintLines(false(size(E)),seglist2segs({x}))>0);
%     
%     aa = (paintLines(false(size(E)),seglist2segs({candidates{21}})));
%     figure,imagesc(aa+bw)
%     
%     montage2(ims_)
    skinTransition = squeeze(mean(mean(bsxfun(@times,ims_,normaliseskinprob))));
%     montage2(bsxfun(@times,ims_,normaliseskinprob))
    
     R = [contourLengths/size(E,1);...
            ucmStrengths;...
            row(pMean(:,1))/size(E,1);...
            verticality';...
            skinTransition';...
            insides;...
            nIntersections];
    if (~debug_)
        allFeats{k} = R;
    end
   
     
    currentScores = R'*[1 1 1 -0.5 0 1 1]';
    currentScores(isnan(currentScores)) = -inf;
    
    [s,is] = sort(currentScores,'descend');            
    currentScores(isnan(currentScores)) = -10;
    if (debug_)
        if (length(ims) >0)
            
%             subplot(2,3,4); imagesc(subUCM);axis image
            subplot(2,3,3),montage2(cat(3,ims{is}))
            %             subplot(2,3,6),montage2(repmat(subUCM,[1 1 length(ims)]));
                                               
            %             normaliseskinprob = imresize(normaliseskinprob,[8 8]);
            subplot(2,3,5); imagesc(normaliseskinprob);axis image
            subplot(2,3,6); imagesc(M); axis image;
            
        end
    end
    
    %     candidates = candidates(contourLengths > 60);
    
    if (~isempty(candidates))
        %         subplot(2,3,5); imagesc(E); axis image; hold on;
        %         drawedgelist(candidates,size(E),2,'rand');
        
        % %         for ic = 1:length(candidates)
        % %             ic
        % % %             im = edgelist2image(candidates(ic), size(E));
        % %             [im,allPts] = paintLines(zeros(size(E)),seglist2segs(candidates(ic)));
        % %             clf; imagesc(E+imdilate((im>0),ones(3))); axis image; hold on;
        % % %             clf; imagesc(im2dilate((im>0),ones(3))); axis image; hold on;
        % %
        % %             %                     drawedgelist(candidates(ic),size(E),2,'rand');
        % %             pause
        % %         end
        %         seglist2edgelist
        %         for ic = 1:length(candidates)
        %             ic
        %             clf; imagesc(E); axis image; hold on;
        %
        %             drawedgelist(candidates(ic),size(E),2,'rand');
        %             pause
        %         end
    end
    %
    %
    if (debug_)
        disp('done');
        pause;
    end
end


%% do some tests on all feats...

allScores = -1*ones(size(cur_t));
%   allFeats{k} = [contourLengths/size(E,1);...
%     ucmStrengths;...
%     1-row(pMean(:,1))/size(E,1);...
%     verticality'];

for k = 1:length(allScores)
    curFeats = allFeats{k};
    if (isempty(curFeats))
        continue;
    end
    %     curFeats(4,:) = curFeats(4,:)> 2;
    
    zz = curFeats(3,:);
    zz = exp(-(zz-.5).^2/1);
%     zz(zz>.7) = 0;
%     zz(zz<.3) = 0;
    curFeats(3,:) = zz;
    pp = curFeats'*[1 1 1 -0.5 0 1]';
    pp(isnan(pp)) = -inf;
    allScores(k) = max(pp);
end

face_comp = [imageData.train.faceLandmarks.c]';

allScores = allScores + 1*ismember(face_comp,6:11);
%  (bbb(:,3)-bbb(:,1))/100
bbb = imageSet.faceBoxes;

[prec,rec,aps,T] = calc_aps2(allScores,cur_t);
%%
% %%
% plot(rec,prec)
%
% [s,is] = sort(allScores,'descend');
% [v,iv] = sort(allScores,'descend');
% for k = 1:length(is)
%
% end

allScores(isinf(allScores)) = -inf;
allScores(isnan(allScores)) = -inf;
% load newScores
%%

newScores2 = max(newScores(:),5*allScores(:));
[prec,rec,aps,T] = calc_aps2(newScores2,imageData.test.labels);
% [v,iv] = sort(newScores2,'descend');