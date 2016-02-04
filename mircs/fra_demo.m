%% 28/7/2014
% predict well the locations of action objects.
% function fra_demo(conf)
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    initialized = true;
    roiParams.infScale = 3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    normalizations = {'none','Normalized','SquareRoot','Improved'};
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV/'));
    initialized = true;
end
if (0)
    load fra_db;
    % boxLocations = cell(1,length(classes));
    % all_polys = cell(1,length(classes));
    roiParams.infScale = 1.5;
    roiParams.absScale = 200;
    
    % obj_box_normalized = {};
    %
    % for iClass = 1:length(classes)
    %     class_boxes = {};
    for iImage=1:length(fra_db)
        curImageData = fra_db(iImage);
        if ~(curImageData.isTrain),continue,end
        %         if ~(curImageData.isTrain && curImageData.classID == iClass),continue,end;
        iImage
        % obtain image
        roiParams.centerOnMouth = true;
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
        rois = rois([rois.id]==3);
        if isempty(rois),continue,end
        curPoly = {rois.poly};
        curBoxes = cellfun2(@pts2Box,curPoly);
        curBoxes = cat(1,curBoxes{:});
        imbb = [1 1 size(I,2),size(I,1)];
        curBoxes = BoxIntersection(curBoxes,imbb);
        [~,~,a] = BoxSize(curBoxes);
        curBoxes = curBoxes(a>0,:);
        f = dsize(I,[2 1 2 1]);
        fra_db(iImage).obj_box_normalized = curBoxes./repmat(f,size(curBoxes,1),1);
        
        % obtain image
        %         segData = load(j2m('~/storage/fra_face_seg',curImageData));segData = segData.res;
        %         candidates = segData.cadidates;
    end
    % boxLocations{iClass} = cat(1,class_boxes{:});
    % end
    boxLocations = cell(1,length(classes));
    for t = 1:length(classes)
        boxLocations{t} = cat(1,fra_db(isTrain & [fra_db.classID]==t).obj_box_normalized);
    end
    allBoxes = cat(1,boxLocations{1:4});    
%     allBoxes = cat(1,boxLocations{3});    
    % add the flips of all boxes
    allBoxes = [allBoxes;[1- allBoxes(:,1),allBoxes(:,2),1-allBoxes(:,3),allBoxes(:,4)]];    
    allBoxCenters = boxCenters(allBoxes);
    %     allBoxCenters = allBoxCenters;[1-allBoxCenters(:,1),allBoxCenters(:,2)]
end


collectGTSegments = false;

if (collectGTSegments)
    true_occlusion_patterns = {};
    false_occlusion_patterns = {};
    %     [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
    for iImage=1:length(fra_db)
        curImageData = fra_db(iImage);
        if (~curImageDsata.isTrain),continue,end;
        iImage
        % obtain image
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
        rois = rois([rois.id]==3);
        if isempty(rois),continue,end
        curPoly = {rois.poly};
        curBoxes = cellfun2(@pts2Box,curPoly);
        curBoxes = cat(1,curBoxes{:});
        [~,~,a_before] = BoxSize(curBoxes);
        imbb = [1 1 size(I,2),size(I,1)];
        curBoxes = BoxIntersection(curBoxes,imbb);
        [~,~,a] = BoxSize(curBoxes);
        goods = a./a_before>.5;
        if (none(goods))
            continue;
        end
        curPoly = curPoly(goods);
        gt_mask = cellfun2(@(x)poly2mask2(x,size2(I)), curPoly);
        %% get facial landmarks
        landmarks = load_fra_landmarks(curImageData);
        if (isempty(landmarks))
            clf;imagesc2(I);
            disp('no faces found.');
            %             pause
            continue
        end
        keepAll  =true;
        [pattern_gt,regions_gt,face_mask,mouth_mask,face_poly,mouth_poly] = getOcclusionPattern_2_new(conf,I,landmarks,gt_mask,keepAll);
        true_occlusion_patterns{iImage} = pattern_gt;
        
        segData = load(j2m('~/storage/fra_face_seg',curImageData));segData = segData.res;
        candidates = segData.cadidates;
        U = max(boxesOverlap(candidates.bboxes,curBoxes),[],2);
        sel_ = vl_colsubset(row(find(U < .1)),100);
        regions = squeeze(mat2cell2(candidates.masks(:,:,sel_),[1,1,length(sel_)]))';
        keepAll = true;
        [occlusionPatterns,regions,face_mask,mouth_mask,face_poly,mouth_poly] = getOcclusionPattern_2_new(conf,I,landmarks,regions,keepAll);
        false_occlusion_patterns{iImage} = occlusionPatterns;
        
    end
    save ~/storage/misc/training_occlusion_patterns.mat true_occlusion_patterns false_occlusion_patterns
    
end

%
if (0)
    %% train a classifier for occlusion patterns...
    truePatterns = cellfun2(@occPatternsToMat,true_occlusion_patterns);
    truePatterns = cat(2,truePatterns{:});
    falsePatterns = cellfun2(@occPatternsToMat,false_occlusion_patterns);
    falsePatterns = cat(2,falsePatterns{:});
    
    X = [truePatterns';falsePatterns'];
    Y = [ones(1,size(truePatterns,2)),-ones(1,size(falsePatterns,2))];
    
    [ii,jj] = find(isnan(X));
    
    
    B = fitensemble(X(:,1:12),Y,'RUSboost',500,'Tree');
    [y,xoe] = predict(B,X);
    
    
    % show some predictions on real test images.
    %%
    
    for iImage=517:length(fra_db)
        curImageData = fra_db(iImage);
      
        iImage
        % obtain image
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
        
        
        % get facial landmarks
        landmarks = load_fra_landmarks(curImageData);
        if (isempty(landmarks))
            clf;imagesc2(I);
            disp('no faces found.');
            %             pause
            continue
        end
        segData = load(j2m('~/storage/fra_face_seg',curImageData));segData = segData.res;
        candidates = segData.cadidates;
        regions = squeeze(mat2cell2(candidates.masks,[1,1,size(candidates.masks,3)]));
        keepAll = true;
        [occlusionPatterns,regions,face_mask,mouth_mask,face_poly,mouth_poly] = getOcclusionPattern_2_new(conf,I,landmarks,regions,keepAll);
        curPatterns = occPatternsToMat(occlusionPatterns);
        [y,xoe] = predict(B,curPatterns(1:12,:)');
        %         scores = xoe(:,1);./xoe(:,2);
        %%
        %     scores = -xoe(:,2)+xoe(:,1);
        %            scores = y;
        %         sel_ = find(probs>.05);
        
        %         probs = probs(sel_);
        %         candidates.masks = candidates.masks(:,:,sel_);
        z = zeros(size2(I));
        nCandidates = size(candidates.masks,3);
        %         bboxes = bboxes(sel_,:);
        for iMask = 1:nCandidates
            z = z+candidates.masks(:,:,iMask)*scores(iMask);
        end
        
        %         figure,imagesc2(z);
        figure;clf; imagesc(sc(cat(3,z,im2double(I)),'prob'));
        
        %figure,plot((xoe(:,1)./xoe(:,2)))
    end
    %%
    
    % figure,plot(xoe);
    
    % plot(resubLoss(B))
    %%
    % close all
    
end
newSegDataDir = '~/storage/fra_face_seg_boxes';
% ensuredir(newSegDataDir);
% % % for t = 1:length(fra_db)
% % %     t
% % %     newFileName = j2m(newSegDataDir,fra_db(t));
% % %     if (exist(newFileName,'file'))
% % %         continue
% % %     end
% % %     curImageData = fra_db(t);
% % %     segData = load(j2m('~/storage/fra_face_seg',curImageData));segData = segData.res;
% % %     candidates = segData.cadidates;
% % %     bboxes = candidates.bboxes;
% % %     save(newFileName,'bboxes');
% % % end
%%
for iImage=1:1:length(fra_db)
    roiParams.infScale = 1.5;
    roiParams.absScale = 200;
    curImageData = fra_db(iImage);
    if (curImageData.classID~=1),continue,end
    if (isTrain(iImage)),continue,end
    iImage
    % obtain image
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);    
    % get the object prediction...    
    resPath = fullfile('~/storage/s40_fra_box_pred_small_extent',[fra_db(iImage).imageID '_' 'obj' '.mat']);
    L = load(resPath);    
    boxFrom = L.roiBox;
    boxTo = [1 1 fliplr(size2(L.pMap))];   
    I_big = getImage(conf,curImageData);
    T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');    
    res = imtransform(L.pMap,T,'XData',[1 size(I_big,2)],'YData',[1 size(I_big,1)],'XYScale',1);
    res = normalise(res);
    %figure,imagesc2(res)
% % %     clf,imagesc2(sc(cat(3,res,im2double(I_big)),'prob_jet'));
% % %     pause;
    %figure,imagesc2(I_big); plotBoxes(L.roiBox)
    
    %U = getImage(conf, L.roiBox
    obj_bbox = L.boxes_orig(1,1:4); % translate to coordinate system of roiBox
    obj_bbox = (obj_bbox-roiBox([1 2 1 2]))*scaleFactor;
%     gt_mask = poly2mask2(box2Pts(obj_bbox),size2(I));
%     gt_mask = cropper(gt_mask,round(mouth_bbox));    
    % get the bounding boxes
    f = dsize(I,[2 1 2 1]);
    
    
    
    
    useSelectiveSearch=true;
    if (useSelectiveSearch) 
        
        
        
        bboxes =  selectiveSearchBoxes(I);
        bboxes = cat(1,bboxes{:});
        bboxes = BoxRemoveDuplicates(bboxes);
        
        
        % transform between the two windows...(that of selective search and
        % that of the 1.5, 200 extent)
% % % % %         L = load(j2m('~/storage/s40_fra_selective_search',curImageData));
% % % % %         [~,roiBox_1,I1,scaleFactor1] = get_rois_fra(conf,curImageData);
% % % % %         bboxes = L.res.boxes;                
% % % % % %             
% % % % % %         for t = 1:50
% % % % % %             clf; imagesc2(I1);             
% % % % % %             plotBoxes(bboxes(t:100:end,:));
% % % % % %             pause;continue;
% % % % % %         end
% % % % % %         
% % % % %         bboxes_orig = bboxes/scaleFactor1+repmat(roiBox_1([1 2 1 2]),size(bboxes,1),1);
% % % % %         
% % % % % %       ovp_1 = boxesOverlap(bboxes_orig,roiBox);
% % % % %         
% % % % %         bboxes = (bboxes_orig-repmat(roiBox([1 2 1 2]),size(bboxes,1),1))*scaleFactor;
% % % % %         badBoxes = any(bboxes < 1,2) | bboxes(:,3)>size(I,2) | bboxes(:,4)>size(I,1);
% % % % %         bboxes(badBoxes,:) = [];
%         bb1 = [90 30 115 59];
%         ovps = boxesOverlap(bboxes,bb1);
%         ovp_t = .1;
%         for t = 1:length(ovps)
%             if (ovps(t)>ovp_t)
%                 clf; imagesc2(I); hold on; plotBoxes(bboxes(t,:));
%                 pause
%             end
%         end

        
        
        
%         figure,hold on;,plotBoxes(allBoxes)
        
%          clf; imagesc2(I);   hold on; plotBoxes(allBoxes*
% % %        
%         for t = 1:50
%             clf; imagesc2(I);             
%             plotBoxes(bboxes(t:100:end,:));
%             pause;continue;
%         end
%         
        nCandidates = size(bboxes,1);
        bboxes_orig = bboxes;
        bboxes = bboxes./repmat(f,nCandidates,1);
        
       
        %     II = getImage(conf,curImageData);
%     figure(1);clf; imagesc2(II);
%     plotBoxes(bboxes_orig(1:100:end,:));
%     figure(2),imagesc2(I1);
%     plotBoxes(bboxes(1:100:end,:));
%     pause; continue;                       
%     bboxes
    else
        %boxes = L.res.boxes
        newFileName = j2m(newSegDataDir,fra_db(iImage));
        load(newFileName);        
        bboxes = BoxRemoveDuplicates(bboxes);
        nCandidates = size(bboxes,1);
        bboxes_orig = bboxes;        
        bboxes = bboxes(:,[2 1 4 3])./repmat(f,nCandidates,1);
    end
    
         
    if (0)        
    useDistance = true;
    if (useDistance)
        D = l2(bboxes,allBoxes).^.5;
        [D,ID] = sort(D,2,'ascend');
        knn = 100;
        sig = .05;
        probs = mean(exp(-D(:,1:knn)/sig),2);
        sel_ = find(probs>0.005);
    else
        D = boxesOverlap(100*bboxes,100*allBoxes);
        sig2 = 1;
        D = 1-D;
        [D,ID] = sort(D,2,'ascend');
        knn = 1;
        sig2 = .1;
        probs = mean(exp(-D(:,1:knn)/sig2),2);
        sel_ = find(probs>0.01);
%         probs = ones(size(probs));
    end 
    else %use the ann stuff..
        res = cropper(res,roiBox);        
        res = imResample(res,size2(I));
        res = normalise(res).^2;
        [bboxes_orig,bads] = clip_to_image(round(bboxes_orig),res);
        bboxes_orig = bboxes_orig(~bads,:);
        probs = sum_boxes(res,bboxes_orig);
        areas = sum_boxes(ones(size2(res)),bboxes_orig);
        probs = probs./areas;
        
%         [r,ir] = sort(probs,'descend');
%         for k = 1:500
%             k
%             clf; 
%             subplot(1,2,1); imagesc2(sc(cat(3,normalise(res),im2double(I)),'prob_jet'));
%             subplot(1,2,2); imagesc2(sc(cat(3,normalise(res),im2double(I)),'prob_jet'));
%             plotBoxes(bboxes_orig(ir(k),:));
%             drawnow;pause(.01)
%         end
        
        sel_ =probs>.7;
    end
%     probs = ones(size(probs));    
    % another probability : just by overlap.
%     ovp = boxesOverlap                
    probs = probs(sel_);
    
    bboxes = bboxes(sel_,:);
    
    z = computeHeatMap(I,[bboxes_orig(sel_,:),probs],'sum');
    z = z/max(z(:));
%     z = z.^2;
%     z = z>.7;
    clf; subplot(1,3,1);imagesc2(sc(cat(3,z,im2double(I)),'prob_jet'));   
    %subplot(1,2,2); imagesc(z); axis image; colormap('jet'); colorbar
    subplot(1,3,2); imagesc2(I);
    subplot(1,3,3); imagesc2(sc(cat(3,normalise(res),im2double(I)),'prob_jet'));
    drawnow ;pause
    
    
    
    continue;
    % get facial landmarks
    load(j2m('~/storage/fra_landmarks',curImageData.imageID));
    landmarks = res.landmarks;
    scores = -inf(size(landmarks));
    for t = 1:length(landmarks)
        if (~isempty(landmarks(t).s))
            scores(t) = landmarks(t).s;
            if (t> 3)
                scores(t) = scores(t)+10;
            end
        end
    end
    [m,im] = max(scores);
    if (isinf(m))
        clf;imagesc2(I);
        disp('no faces found.');
        pause
        continue
    end
    
    figure(2);clf;imagesc2(I);
    curPoly = cellfun2(@mean,landmarks(im).polys);
    curPoly  = cat(1,curPoly{:});
    plotPolygons(curPoly,'r.');
    
    nCandidates = size(candidates.masks,3);
    %     v = vl_colsubset(1:nCandidates,5);
    %     displayRegions(I,squeeze(mat2cell2(candidates.masks(:,:,v),[1,1,length(v)])),[],.1);
    
    % occlusion patterns / features
    regions = squeeze(mat2cell2(candidates.masks,[1,1,nCandidates]));
    [occlusionPatterns,regions,face_mask,mouth_mask,face_poly,mouth_poly] = getOcclusionPattern_2_new(conf,I,landmarks(im),regions);
    continue
    imageData.faceScore=0;
    %[M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,imageData,1.5,1);
    faceLandmarks = landmarks(im);
    curLandmarks = faceLandmarks;
    occlusionPattern.mouth_poly = mouth_poly;
    occlusionPattern.face_poly = face_poly;
    occlusionPattern.occlusionPatterns = occlusionPatterns;
    occlusionPattern.regions = regions;
    occlusionPattern.face_mask = face_mask;
    occlusionPattern.mouth_mask = mouth_mask;
    occlusionPattern.faceBox = pts2Box(face_poly);
    clear occluionData;
    [occlusionData.occludingRegions,occlusionData.occlusionPatterns,occlusionData.rprops] = getOccludingCandidates_2(I,occlusionPattern);
    
    occScores = score_occluders_new_2(I,occlusionPattern,faceLandmarks,occlusionData.occludingRegions,occlusionData.occlusionPatterns);
    %     displayRegions(I,occlusionData.occludingRegions,[],.5)
    edit score_occluders_new.m
    displayRegions(I,occlusionData.occludingRegions,occScores,0,5)
    
    %     pause
end
%     pause
% end