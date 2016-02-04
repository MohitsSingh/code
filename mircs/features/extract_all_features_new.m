function [regionFeats,imageFeats,selected_regions] = extract_all_features_new(conf,imgData,params,imageFeats,selected_regions)
%Aggregates multiple feature types for images. Some features may have been
% precomputed

% obtain:
% 1. image
% 2. facial keypoints
% 3. segmentation
% 4. saliency
% 5. object mask
% 6. shape probability prediction
% 7. line segment features (to server as additional candidates.

regionFeats = struct('label',{},'mask',{},'geometricFeats',{},'shapeFeats',{},'meanSaliency',{},'meanProb',{});
if (isfield(params,'prevStageDir') && ~isempty(params.prevStageDir))
    load(j2m(params.prevStageDir,imgData));
    selected_regions = toKeep;
end

if (isfield(params,'get_gt_regions'))
    params.get_gt_regions = true;
end

if (isfield(params,'externalDB') && params.externalDB)
    is_in_fra_db = -1;
else
    is_in_fra_db = imgData.indInFraDB~=-1;
end

% if this is a training image, use the ground-truth data
imgData = switchToGroundTruth(imgData);
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,imgData,params.roiParams);
[I_orig,I_rect] = getImage(conf,imgData);
imageFeats.I = I;

% first , make sure that the image contains enough of the action object. If
% not, consider it invalid (except for test mode, where we cannot use this
% information)
[gt_object_masks,bads] = getGtPolygons(I,rois);
if isempty(bads) || all(bads)
    if (~params.testMode)
        imageFeats.valid = false;
        return;
    end
end

imageFeats.valid = true;
gt_object_masks = gt_object_masks(~bads);

% facial keypoints
disp('extracting facial landmarks...');
if (~imgData.isTrain || (imgData.isTrain && ~is_in_fra_db)) % if a test image, read the predicted keypoints
    landmarksDir = fullfile(fullfile(params.dataDir,'facial_keypoints'));
    landmarksCachePath = j2m(landmarksDir,imgData);
    
    if (exist(landmarksCachePath,'file'))
        load(landmarksCachePath);
    else
        landmarkInit = params.landmarkParams;
        faceBox = rois(1).bbox;
        landmarkInit.debug_ = false;
        bb = round(faceBox(1,1:4));
        [kp_global,kp_local,kp_preds] = myFindFacialKeyPoints(conf,I,bb,landmarkInit.XX,...
            landmarkInit.kdtree,landmarkInit.curImgs,landmarkInit.ress,landmarkInit.ptsData,landmarkInit);
        ensuredir(landmarksDir);
        save(landmarksCachePath,'kp_global','kp_local','kp_preds');
    end
    
    kp_centers = boxCenters(kp_preds);
    %     kp_centers = transformToLocalImageCoordinates(kp_centers,scaleFactor,roiBox);
    kp_preds = [inflatebbox([kp_centers kp_centers],[5 5],'both',true) kp_preds(:,end)];
    nKP = size(kp_centers,1);
else
    [kp_preds,goods] = loadKeypointsGroundTruth(imgData,params.requiredKeypoints);
    confidences = kp_preds(:,3);
    kp_preds = kp_preds(:,1:2);
    kp_preds = kp_preds-repmat(roiBox(1:2),size(kp_preds,1),1);
    kp_preds = kp_preds*scaleFactor;
    kp_centers = kp_preds;
    kp_preds = inflatebbox([kp_preds kp_preds],[5 5],'both',true);
    nKP = size(kp_preds,1);
end

imageFeats.kp_preds = kp_preds;

% showStuff(); return;

iMouth = find(strncmpi('mouth',{rois.name},5));
imageFeats.roiMouth = inflatebbox(rois(iMouth).bbox,[60 40],'both',true);
if (params.skipCalculation)
    showStuff();
    return;
end

imageFeats.kp_confidences = kp_preds(:,end);
gotSelectedRegions = exist('selected_regions','var'); % recompute and get only a subset of the regions?
if (gotSelectedRegions)
    regionSubset = selected_regions;
end
disp('getting segmentation...');

face_seg_dir = fullfile(params.dataDir,'face_seg');
face_seg_path = j2m(face_seg_dir,imgData);
if (~exist(face_seg_path,'file'))
    ensuredir(face_seg_dir);
    [candidates, ucm] = im2mcg(I,'fast',false); % warning, using fast segmentation...
    save(face_seg_path,'candidates','ucm');
else
    load(face_seg_path);
end
candidates.masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);

% check for correct size of candidate masks...this shouldn't happen
% unless some parameter was changed inconsistenty.
if (~all(size2(candidates.masks)==size2(I)))
    warning('need to recompute candidate masks...');
    [candidates, ucm] = im2mcg(I,'fast',false); % 
    save(face_seg_path,'candidates','ucm');
    candidates.masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
end
candidates.masks = squeeze(mat2cell2(candidates.masks,[1 1 size(candidates.masks,3)]));

selected_regions = candidates.masks;
region_scores = candidates.scores;

% heuristically remove obvious none-candidates
selected_regions(region_scores < .1) = [];
region_scores(region_scores < .1) = [];
[selected_regions,toKeep] = removeObviousSegments(selected_regions);
region_scores = region_scores(toKeep);
%     end
ucm = ucm(1:2:end-1,1:2:end-1);
imageFeats.ucm = ucm;
% line segments features

if (params.features.getLineSegs)
    disp('obtaining line segments...');
    %         if (isfield(moreData,'lineSegments'))
    %             lineSegments = moreData.lineSegments;
    %         else
    
    % don't want the elsd for now
    
    %     lineSegments = getELSDLineSegmentFeatures(conf,imgData,I);
    %     lineSegments = clipEdge(lineSegments, roiBox([1 3 2 4]));
    %     lineSegments(all(lineSegments==0,2),:) = [];
    %     lineSegments = lineSegments-repmat(roiBox([1 2 1 2]),size(lineSegments,1),1);
    %     lineSegments = lineSegments*scaleFactor;
    lineSegments = zeros(0,4);
    face_line_seg_dir = fullfile(params.dataDir,'face_line_segments');
    lineSegsPath = j2m(face_line_seg_dir,imgData);
    if (~exist(lineSegsPath,'file'))
        ensuredir(face_line_seg_dir);
        E = edge(im2double(rgb2gray(I)),'canny');
        [edgelist edgeim] = edgelink(E, []);
        segs = [lineseg(edgelist,2),lineseg(edgelist,4),lineseg(edgelist,6)];
        segs = seglist2segs(segs);
        segs = unique(segs,'rows');
        %     x2(I); hold on;plotSegs(segs(:,[2 1 4 3]),'g-')
        lineSegments = [segs(:,[2 1 4 3]);lineSegments];
        save(lineSegsPath,'lineSegments');
    else
        load(lineSegsPath);
    end
    
    imageFeats.lineSegments = lineSegments;
    % add elsd line segments too
    candidatesFromLines = processLinesToCandidates(I,lineSegments)';
    nonZeros = @(x) cellfun(@nnz,x);
    assert(all(nonZeros(candidatesFromLines)));
    assert(all(nonZeros(candidates.masks)));
    selected_regions = [candidatesFromLines;selected_regions];
end

% add the ground-truth regions for evaluation, if there are any
nRegions = length(selected_regions);
nGT = length(gt_object_masks);
region_labels = imgData.classID*ones(nRegions+nGT,1);
selected_regions = [selected_regions(:);gt_object_masks(:)];
is_gt_region = false(size(selected_regions));
is_gt_region(nRegions+1:end) = true;

if (gotSelectedRegions)
    selected_regions = selected_regions(regionSubset);
    region_labels = region_labels(regionSubset);
    is_gt_region = is_gt_region(regionSubset);
end

% calculate overlap of regions with ground-truth, obviously for gt regions
% this will be one
if (isempty(gt_object_masks))
    ovp = zeros(1,length(selected_regions));
else
    ovp = regionsOverlap(gt_object_masks,selected_regions);
    ovp = max(ovp,[],1);
end

% end
% saliency (todo - later load it rather than compute each time)
hasMoreData = exist('moreData','var');
% if (~hasMoreData)
%     imageFeats = struct;
%     imageFeats.imgIndex = imgData.imgIndex;
%     imageFeats.valid = true;
% end

if (isfield(imageFeats,'saliency'))
    sal = imageFeats.saliency.sal;
    %sal = paintSeg(sal,segData.candidates.superpixels);
    sal_bd = imageFeats.saliency.sal_bd;
    sal_global = imageFeats.saliency.sal_global;
    sal_bd_global = imageFeats.saliency.sal_bd_global;
else
    disp('computing saliency...');
    
    saliencyDir = fullfile(fullfile(params.dataDir,'saliency'));
    saliencyPath = j2m(saliencyDir,imgData);
    if (exist(saliencyPath,'file'))
        load(saliencyPath);
    else
        ensuredir(saliencyDir);
        [sal,sal_bd,sal_global,sal_bd_global] = extractSaliencyHelper(conf,imgData,params.saliencyParams);
        sal = imResample(cropper(sal,round(roiBox)),size2(I));
        sal_bd = imResample(cropper(sal_bd,round(roiBox)),size2(I));
        sal_global= imResample(cropper(sal_global,round(roiBox)),size2(I));
        sal_bd_global = imResample(cropper(sal_bd_global,round(roiBox)),size2(I));
        save(saliencyPath,'sal','sal_bd','sal_global','sal_bd_global');
    end
    
    
    %     figure(7); imagesc2(sal_bd);
    imageFeats.saliency.sal = sal;
    imageFeats.saliency.sal_bd = sal_bd;
    imageFeats.saliency.sal_global = sal_global;
    imageFeats.saliency.sal_bd_global = sal_bd_global;
end

if (~isfield(imageFeats,'face_feats'))
    net = params.features.nn_net ;
    
    %     net = init_nn_network();
    imageFeats.face_feats.global = extractDNNFeats(I,net);
    imageFeats.face_feats.mouth = extractDNNFeats(cropper(I,round(makeSquare(imageFeats.roiMouth,true))),net);
    get_full_image = false;
    I_crop = getImage(conf,imgData,get_full_image);
    imageFeats.global_feats = extractDNNFeats(I_crop,net);
end

% object probability prediction
if (~isfield(imageFeats,'predictions'))
    disp('predicting object location...');
    [objPredictionImage,maskPredictionImage] = getPredictionImages(conf,imgData,roiBox);
    maskPredictionImage = imResample(maskPredictionImage,size2(I));
    objPredictionImage = imResample(objPredictionImage,size2(I));
    objPredictionImage = normalise(objPredictionImage).^.5;
    imageFeats.predictions.objPredictionImage = objPredictionImage;
    imageFeats.predictions.maskPredictionImage = maskPredictionImage;
else
    objPredictionImage = imageFeats.predictions.objPredictionImage;
    maskPredictionImage = imageFeats.predictions.maskPredictionImage;
end
% maskPredictionImage = normalise(maskPredictionImage);
% maskPredictionImage = maskPredictionImage.^.5;
if params.segmentation.useGraphCut
    % graph-cut segmentation
    gc_segResult = getSegments_graphCut_2(I,objPredictionImage,[],0);
    gc_segResult = double(repmat(gc_segResult,[1 1 3])).*I;
end

allRegionBoxes = cellfun2(@region2Box,selected_regions);
allRegionBoxes = cat(1,allRegionBoxes{:});

tic_id = ticStatus( 'extracting segment feautures', 1,1);

if (params.features.getSimpleShape)
    simpleShapes = cellfun2(@(x) imResample(im2single(x),[7 7],'bilinear'),selected_regions);
end

resizeFactor = .3;
smallMasks = cellfun2(@(x) imResample(im2single(x),round(size2(x)*resizeFactor),'bilinear')>0,selected_regions);

% create log-polar masks for each of the 21 keypoints.
m = getLogPolarMask(30,6,3);%(10,nTheta,nLayers);

z_shapes = {};zeros([size2(I),nKP]);
nbins = max(m(:));
for iz = 1:nKP
    kp_topleft = round(kp_centers(iz,:)-.5*fliplr(size2(m)));
    kp_bottomright = kp_topleft+fliplr(size2(m))-1;
    T = shiftRegions(m,[kp_topleft,kp_bottomright],I);
    T(T==0) = nbins+1;
    z_shapes{iz} = imResample(T,resizeFactor,'nearest');
end

if (params.features.getAppearance)
    [learnParams,conf] = getDefaultLearningParams(conf,256);
    featureExtractor = learnParams.featureExtractors{1};
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    disp('extracting appearance features...');
    appearanceFeats = featureExtractor.extractFeatures(I,selected_regions,'normalization','Improved');
end

if (params.features.getAppearanceDNN)
    % crop all masks to sub-images
    subImgs = multiCrop2(I,allRegionBoxes);
    dnn_feats = extractDNNFeats(subImgs,params.features.nn_net);
end
% if (params.features.getAppearance)
iFaceBox = find(strncmpi('face',{rois.name},3));
[bbox_feats,B,regionBoxes] = get_bbox_feats(selected_regions,rois(iFaceBox).bbox,size2(I),boxCenters(kp_preds));
for t = 1:length(selected_regions)
    regionFeats(t).label = region_labels(t);
    regionFeats(t).bbox = regionBoxes(t,:);
    regionFeats(t).is_gt_region = is_gt_region(t);
    curMask = selected_regions{t};
    regionFeats(t).gt_ovp = ovp(t);
    regionFeats(t).boxFeats = B(t,:);
    
    if (params.features.getSimpleShape)
        regionFeats(t).simpleShape = simpleShapes{t};
    end
    
    if (params.features.getAppearanceDNN)
        regionFeats(t).dnn_feats = dnn_feats(:,t);
    end
    
    
    if (params.features.getHOGShape)
        bb = region2Box(curMask);
        bb = makeSquare(bb,true);
        bb = inflatebbox(bb,1.1);
        bb_img = cropper(curMask,round(bb));
        bb_img = imResample(im2single(bb_img),[64 64],'bilinear');
        regionFeats(t).HOGShape = col(fhog(bb_img));
    end
    
    %     if (params.features.getHOGShapeRotated)
    %         bb = region2Box(curMask);
    %         bb = makeSquare(bb,true);
    %         bb = inflatebbox(bb,1.1);
    %         bb_img = cropper(curMask,round(bb));
    %         bb_img = imResample(im2single(bb_img),[64 64])
    %         feats(t).HOGShape =
    %     end
    %
    if (params.features.getAppearance)
        regionFeats(t).appearance = appearanceFeats(:,t);
    end
    
    
    if (params.features.getLogPolarShape)
        [logPolarMask,m_vis] = getLogPolarShape(curMask,36);
    else
        logPolarMask = NaN;
    end
    
    if (params.features.getShape)
        r = regionprops(curMask,'Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
        r = r(1);
        shapeFeats = [r.Area^.5;r.Eccentricity;r.MajorAxisLength;r.MinorAxisLength;r.Orientation];
        regionFeats(t).shapeFeats = shapeFeats;
    else
        regionFeats(t).shapeFeats = NaN;
    end
    % % % %     feats(t).shapeFeats = [logPolarMask(:);r.Area^.5;r.Eccentricity;r.MajorAxisLength;r.MinorAxisLength;r.Orientation;r.Solidity];
    
    if (params.features.getGeometry)
        %[y,x] = find(curMask);
        [y,x] = find(smallMasks{t});
        
        % for each keypoint, find the nearest and furthest distance on the mask
        % to this keypoint.
        xy_mask = [x(:) y(:)]/resizeFactor;
        kp_to_mask = l2(kp_centers,xy_mask);
        regionFeats(t).geometricFeats = [min(kp_to_mask,[],2);max(kp_to_mask,[],2)];
    end
    if (params.features.getGeometryLogPolar)
        geometric_feats_log_polar = zeros(nKP,nbins+1);
        S = smallMasks{t}>0;
        for u = 1:size(kp_centers,1)
            curZShape = z_shapes{u};
            accum = zeros(nbins+1,1);
            geometric_feats_log_polar(u,:)=vl_binsum(accum,1,curZShape(S));
        end
        regionFeats(t).geometric_feats_log_polar =  geometric_feats_log_polar(:);
    end
    
    %     feats(t).geometricFeats = [min(kp_to_mask,[],2);max(kp_to_mask,[],2)];
    
    if (params.features.getPixelwiseProbs)
        regionFeats(t).meanSaliency = mean(sal(curMask(:)));
        regionFeats(t).meanBDSaliency = mean(sal_bd(curMask(:)));
        regionFeats(t).meanGlobalSaliency = mean(sal_global(curMask(:)));
        regionFeats(t).meanGlobalBDSaliency = mean(sal_bd_global(curMask(:)));
        regionFeats(t).meanLocationProb = mean(objPredictionImage(curMask(:)));
        regionFeats(t).meanShapeProb = mean(maskPredictionImage(curMask(:)));
    end
    tocStatus(tic_id,t/length(selected_regions));
end

% visualization (for debugging)
if (params.debug)
    showStuff()
end
    function showStuff()
        mm = 3;
        nn = 3;
        ss = 1;
        % if (debug_)
        
        subplotFun = @subplot;
        
        % show keypoints
        figure(1);clf;
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(I);
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(I);title('original + keypoints + object');
        plotBoxes(kp_preds(goods,:));
        plotBoxes(kp_preds(3,:),'color','y','LineWidth',3);
        plotBoxes(kp_preds(4,:),'color','r','LineWidth',3);
        plotBoxes(kp_preds(5,:),'color','c','LineWidth',3);
        % show object
        iObj = find(strncmpi('obj',{rois.name},3));
        
        % % %     if ~isempty(iObj)
        % % %         for ii = 1:length(iObj)
        % % %             plotPolygons(rois(iObj(ii)).poly,'r-','LineWidth',2);
        % % %         end
        % % %     end
        % show segmentation
        warning('showing only part of the stuff...');
        return;
        
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(ucm);title('ucm');
        % show saliency
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(moreData.saliency.sal);title('foreground');
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(moreData.saliency.sal_bd);title('foreground-bd');
        % show object prediction image
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(sc(cat(3,moreData.predictions.objPredictionImage,I),'prob_jet'));title('predicted obj. loc');
        if params.segmentation.useGraphCut
            subplotFun(mm,nn,ss);ss = ss+1;
            imagesc2(gc_segResult);title('graph-cut segmentation');
        end
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(sal_global); title('foreground - global');
        subplotFun(mm,nn,ss);ss = ss+1;
        imagesc2(sal_bd_global);title('foreground - bd - global');
        if (params.features.getLineSegs)
            subplotFun(mm,nn,ss);ss = ss+1;
            imagesc2(I); plotSegs(lineSegments,'g-');
        end
    end
    function [gt_masks,bads] = getGtPolygons(I,rois)
        iObj = find(strncmpi('obj',{rois.name},3));
        gt_masks = {};
        bads = false(size(iObj));
        if ~isempty(iObj)
            for ii = 1:length(iObj)
                curPoly = rois(iObj(ii)).poly;
                Z = poly2mask2(curPoly,size2(I));
                gt_masks{ii} = Z;
                if (nnz(Z)/polyarea(curPoly(:,1),curPoly(:,2)) < .5)
                    % discard only if a small object...
                    if (nnz(Z) < 100) % TODO - check if this is a good heuristic
                        bads(ii) = true;
                        %                         clf; displayRegions(I,Z);
                        %                         continue;
                    end
                end
                %gt_masks{end+1} = Z;
            end
        end
    end
    function [masks,toKeep] = removeObviousSegments(masks)
        %         displayRegions(I,masks(randperm(length(masks))));
        % remove too large regions
        sz = prod(size2(I));
        toKeep = 1:length(masks);
        areas = cellfun(@nnz,masks);
        maxArea = .4;
        minAreaPix = 30;
        masks(areas/sz > maxArea | areas < minAreaPix) = [];
        toKeep(areas/sz > maxArea | areas < minAreaPix) = [];
        %         displayRegions(I,masks)
        % remove regions for which the ratio of pixels taken from the
        % border of the image is too large
        Z = addBorder(false(size2(I)),1,1);
        totalBorderElements = nnz(Z);
        maskBorderElements = cellfun(@(x) sum(x(Z)),masks);
        maxBorderElements = .2;
        masks(maskBorderElements/totalBorderElements>=maxBorderElements) = [];
        toKeep(maskBorderElements/totalBorderElements>=maxBorderElements) = [];
        maskBorderElements(maskBorderElements/totalBorderElements>=maxBorderElements) = [];
        
        % remove region for which the ratio of border elements w.r.t the
        % total region perimeter is too large
        % calculate a lower bound on perimeter using an equivalent circle
        equivCircleDiameter = cellfun(@(x) 2*(nnz(x)*pi)^.5, masks);
        maxSelfBorder = .5;
        
        masks(maskBorderElements./equivCircleDiameter > maxSelfBorder) = [];
        toKeep(maskBorderElements./equivCircleDiameter > maxSelfBorder) = [];
        
    end

    function [objPredictionImage,maskPredictionImage] = getPredictionImages(conf,imgData,roiBox)
        if (is_in_fra_db)
            resDir = '~/storage/s40_fra_box_pred_2014_09_17';
            resPath = fullfile(resDir,[imgData.imageID '_' 'obj' '.mat']);
            L_obj = load(resPath);
        else
            obj_pred_dir = '~/storage/s40_obj_prediction';
            objPredPath = j2m(obj_pred_dir,imgData);
            
            if (exist(objPredPath,'file'))
                load(objPredPath,'L_obj');
            else
                ensuredir(obj_pred_dir);
                %%%%%%%%%%%%%%%%%%%%%%%%%
                params.objPredData.debugParams.debug = false;
                L_obj = predictObj(conf,imgData,params.objPredData);
                save(objPredPath,'L_obj');
                %%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
        f =  @(x)  cropper(normalise(transformBoxToImage(I_orig, x, L_obj.roiBox, false)),roiBox).^4;
        objPredictionImage = f(L_obj.pMap);
        maskPredictionImage = f(L_obj.shapeMask);
        
    end
    function [sal,sal_bd,sal_global,sal_bd_global] = extractSaliencyHelper(conf,imgData,saliencyParams)
        curRoiParams.infScale = 1.5;
        curRoiParams.absScale = 100;
        [curRois,curRoiBox,I_sal] = get_rois_fra(conf,imgData,curRoiParams);
        [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I_sal),saliencyParams);
        sal = transformBoxToImage(I_orig, sal, curRoiBox, false);
        sal_bd = transformBoxToImage(I_orig, 1-sal_bd, curRoiBox, false);
        
        %         I_orig = getImage(conf,imgData);
        [sal_global,sal_bd_global,resizeRatio] = extractSaliencyMap(im2uint8(I_orig),saliencyParams);
        sal_global = cropper(imResample(sal_global,size2(I_orig)),round(curRoiBox));
        sal_bd_global = cropper(imResample(sal_bd_global,size2(I_orig)),round(curRoiBox));
        sal_global = transformBoxToImage(I_orig, sal_global, curRoiBox, false);
        sal_bd_global = transformBoxToImage(I_orig, 1-sal_bd_global, curRoiBox, false);
    end
end
