
if (~exist('toStart','var'))
    initpath;
    config;
    load lipData.mat;
    
    initLandMarkData;
    % %     close all;
    clf;
    
    conf.class_subset = conf.class_enum.DRINKING;
    %     [action_rois,true_ids] = markActionROI(conf,roiPath);
    
    % load train-hands
    load ~/storage/hands_s40/top_bbs_train.mat
    top_bbs_train = top_bbs_train(train_dets.cluster_locs(:,11));
    top_bbs_train = top_bbs_train(train_face_scores >= min_train_score);
        
    % clear st;
    
    % root: detect face at given pose.
    % node 1: detect lips / not detect lips.
    % node 2: detect stra
    % w / not detect straw.
    
    % root is given, now proceed to detect lips. use lips from faces with good
    % scores.
    
    face_comp_train = [faceLandmarks_train_t.c];
    face_comp_test = [faceLandmarks_test_t.c];
    % stage 1: assume lips are detected correctly, find straight line segments
    % from lips.
    
    % addpath('/home/amirro/code/SGE');
    % sge_gpb(lipImages_train,'~/storage/gbp_train');
    % sge_gpb(lipImages_test,'~/storage/gbp_test');
    %
    % baseDir ='~/storage/gbp_train';
    % calculate gbp for all images.
    % calculateGpbParaller('~/storage/gbp_train/00001.tif');
    
    % find straight line segments.
    
    lipImages_test_2 = multiCrop(conf,lipImages_test,[], [100 NaN]);
    lipImages_train_2 = multiCrop(conf,lipImages_train,[], [100 NaN]);
    
    clusters = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','multilips',...
        'override',false);
    
    % learn the appearance of cups....
    conf.class_subset = conf.class_enum.DRINKING;
    % roiPath = '~/storage/action_rois';
    %     [action_rois,true_ids] = markActionROI(conf,roiPath);
    
    
    roiPath = '~/storage/cup_rois';
    [action_rois,true_ids] = markActionROI(conf,roiPath);
    
    % feats_train = getBOWFeatures(conf,model,set1,[],[]);
    
    action_rois_s = makeSquare(action_rois);
    % figure,hold on; plotBoxes2(action_rois_s(:,[2 1 4 3]));axis equal;
    
    % % % % % % % conf.not_crop = true;
    % % % % % % % action_images = multiCrop(conf,true_ids,round(action_rois_s),[64 64]);
    % % % % % % % mImage(action_images);
    
    
    
    % % % % % % %
    % % % % % % %
    % % % % % % % conf.not_crop = false;
    % % % % % % % action_images_flip = flipAll(action_images);
    % % % % % % % action_images = [action_images,action_images_flip];
    % % % % % % % % do some clustering...
    % % % % % % % conf.features.winsize  = [10 10];
    % % % % % % %
    % % % % % % %
    conf.features.vlfeat.cellsize = 4;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.clustering.num_hard_mining_iters = 12;
    conf.features.winsize = [10 10 31];
    conf.detection.params.detect_keep_threshold = -1;
    
    % % % % % % % x = imageSetFeatures2(conf,action_images,true,[40 40]);
    % % % % % % % conf.clustering.num_hard_mining_iters = 5;
    % % % % % % % [C,IC] = vl_kmeans(x,10,'NumRepetitions',10);
    % % % % % % %
    % % % % % % % [cup_clusters,ims]= makeClusterImages(action_images,C,IC,x,'drinking_action_images');
    % % % % % % % for k = 1:length(cup_clusters)
    % % % % % % %     cup_clusters(k).cluster_samples = cup_clusters(k).cluster_samples(:,1:3);
    % % % % % % % end
    % % % % % % % conf.max_image_size = 128;
    % % % % % % % cup_clusters_train = train_patch_classifier(conf,cup_clusters,getNonPersonIds(VOCopts),'suffix',...
    % % % % % % %     'cup_clusters','override',true,'C',.1);
    % % % % % % % conf.max_image_size = inf;
    % % % % % % %
    % % % % % % % for k = 1:length(cup_clusters_train)
    % % % % % % %     subplot(1,2,1);imshow(showHOG(conf,cup_clusters_train(k).w));
    % % % % % % %     subplot(1,2,2); imshow(ims{k});
    % % % % % % %     pause;
    % % % % % % % end
    
    % qq_t = applyToSet(conf,cup_clusters_train,imageIDs(t_test),[],'cups_train','toSave',true,'visualizeClusters',...
    %     true,'uniqueImages',true,'override',true);
    
    qq_test = applyToSet(conf,clusters,lipImages_test_2,[],'multilips_test','toSave',true,'visualizeClusters',...
        false,'uniqueImages',false,'override',false);
    
    qq_train = applyToSet(conf,clusters,lipImages_train_2,[],'multilips_train','toSave',true,'visualizeClusters',...
        false,'uniqueImages',false,'override',false);
    
    check_test = true;
    clear test_faces_2;
    clear get_full_image;
    if (check_test)
        suffix = 'test';
        cur_t = t_test;
        cur_set = lipImages_test;
        cur_dir = '~/storage/gbp_test/';
        ucms = getAllUCM(cur_dir,numel(cur_t));
        
        for k = 1:length(ucms)
            
            ucms{k}=single(ucms{k});
        end
        cur_comp = face_comp_test;
        cur_face_scores = test_faces_scores_r;
        [regions,regionInds,imageInds,allProps] = extractProperties(cur_set,ucms,qq_test,suffix);
        imageIDs = test_ids_r;
        curLipBoxes = lipBoxes_test_r_2;
        curFaces = test_faces;
        locs_r = test_locs_r;
    else        
        % load('~/storage/train_gpbs.mat');
        cur_t = t_train
        cur_dir = '~/storage/gbp_train/'
        cur_set = lipImages_train;
        suffix = 'train';
        ucms = getAllUCM(cur_dir,numel(cur_t));
        cur_comp = face_comp_train;
        cur_face_scores = train_faces_scores_r;
        [regions,regionInds,imageInds,allProps] = extractProperties(cur_set,ucms,qq_train,suffix);
        imageIDs = train_ids_r;
        curLipBoxes = lipBoxes_train_r_2;
        curFaces = train_faces;
        locs_r = train_locs_r;
    end
    
    clear qq_test;
    clear qq_train;
    clear lipImages_test_2;
    clear lipImages_train_2;
    clear lipImages_test;
    clear lipImages_train;
    clear ucm;
    % 1 area
    % 2 ecce
    % 3 kk
    % 4 maj. length
    % 5 max. pts. x
    % 6 max. pts. y
    % 7 min. pts. x
    % 8 min. pts. y
    % 9 mean intensity
    % 10 minor axis
    % 11 orientation
    % 12 solidity
    % 13 startPts
    % 14 z_in
    % 15 z_out
    
    % sort regions by top y point.
    %%yy = allProps(13,:);
    
    area_ = 1;
    ecce_ = 2;
    kk_ = 3;
    majAxis_ = 4;
    meanIntensity_ = 9;
    minAxis_ = 10;
    orientation_ = 11;
    solidity_ = 12;
    startPts_ = 13;
    z_in_ = 14;
    z_out_ = 15;
    
    mmm = {};
    
    % pack regions into cells....
    regionCells = {};
    f = [0;find(diff(regionInds));length(regionInds)];
    imageInds_ = zeros(1,size(allProps,2));
    for k = 1:length(f)-1
        if (mod(k,1000)==0)
            disp(100*k/length(f));
        end
        regionCells{k} = regions(f(k)+1:f(k+1));
        imageInds_(k) = imageInds(f(k)+1);
    end
    disp('done');
    
    load('~/code/kmeans_4000.mat','codebook');
    model.numSpatialX = [2];
    model.numSpatialY = [2];
    model.vocab = codebook;
    load w.mat;
    toStart = 0;
end

%%
close all;
nn = 0;
% d = load('~/code/3rdParty/kmeans_4000.mat');

theta_cond = abs(allProps(orientation_,:)) > 15;
checkedImages = false(size(cur_t));
yy = 100*allProps(startPts_,:) +...
    (allProps(solidity_,:)>.5) +...
    (allProps(kk_,:)>.1) +...
    .001*(allProps(majAxis_,:)) +...
    (allProps(minAxis_,:) < 20) +...
    (allProps(z_in_,:)) -...
    (allProps(z_out_,:))+...
    (allProps(area_,:) < 1000) +...
    +(allProps(area_,:) > 50) +...
    +1*(cur_face_scores(imageInds_)>-.7) +...
    (allProps(meanIntensity_,:)) +...
    1*theta_cond +...
    0*.01*abs(allProps(orientation_,:)) + ...
    1*ismember(cur_comp(imageInds_),[5:9]);
%   ( > -.8);
%
% zout

[r,ir] = sort(yy,'descend');
mmm = {};
curScore = {};
maskedImages = {};
chosenInds = {};
chosenImages = {};
conf.detection.params.detect_min_scale = .1;
tt = [];
% ir = randperm(length(ir));

%for k = 62:length(ir)
% for k = 170:length(ir)
% for k = 1:length(ir)
% main loop
for k = 1:length(ir)
    
    % %     close all;
    k
    curInds = regionCells{ir(k)};
    imageInd = imageInds_(ir(k));
    currentID = imageIDs{imageInd};
    if (checkedImages(imageInd))
        continue;
    end
    % % %
        if(~cur_t(imageInd))
            continue;
        end
    % % %
    % % %     if (~ismember(imageInd,strawInds_abs))
    % % %         continue;
    % % %     end
    
    % extract the current sub-image.
    
    nn = nn+1;
    curScore{nn} = r(k);
    chosenInds{nn} = ir(k);
    tt = [tt,cur_t(imageInd)];
    checkedImages(imageInd) = true;
    chosenImages{nn} = imageInd;
    curImage = cur_set{imageInd};
    toMask = zeros(2*dsize(curImage,1:2));
    toMask(curInds) = 1;
    %     toMask = imdilate(toMask,ones(7));
    maskedImages{nn} = bsxfun(@times,im2double(imresize(curImage,[100 100],'bilinear')),toMask);
    %     curImage = train_faces{imageInd};
    maskedImages{nn} = imresize(maskedImages{nn},.5,'bilinear');
    mmm{nn} = curImage;
    if (nn >= 200)
        break;
    end
    
    % follow the segment along the direction.
    
    %     imagesc(curUCM)
    
    curImage = imresize(curImage,2,'bilinear');
    startPoint = allProps(7:8,ir(k));
    endPoint = allProps(5:6,ir(k));
    curLipRect = curLipBoxes(imageInd,:);
    faceImage = curFaces{imageInd};
    
    ori =  allProps(orientation_,ir(k));
    % flip the x of the mouth location if necessary
    curTitle = '';
    curFaceBox = locs_r(imageInd,:);
    if (curFaceBox(:,conf.consts.FLIP))
        curTitle = 'flip';
        curLipRect = flip_box(curLipRect,[128 128]);
        startPoint(1) = 100-startPoint(1);
        endPoint(1) = 100-endPoint(1);
        ori = -ori;
        
    end
    vec = endPoint-startPoint;
    vec_ori = [cosd(ori),sind(ori)];
    
    if (dot(vec_ori,vec) < 0)
        vec_ori = -vec_ori;
    end
    %     vec = norm(vec)*vec_ori;
    
    conf.not_crop = false;
    [fullImage,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
    conf.detection.params.detect_keep_threshold = -1;
    conf.detection.params.detect_exemplar_nms_os_threshold = 0;
    conf.detection.params.detect_max_windows_per_exemplar = 20;
    conf.detection.params.detect_add_flip = 0;
    
    conf.not_crop = true;
    [f2,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
    
    faceBoxShifted = curFaceBox(1:4) + [xmin ymin xmin ymin];
    %     imshow(f2); hold on; plotBoxes2(faceBoxShifted([2 1 4 3]));
    conf.not_crop = false;
    
    % now get the "full" image
    %         continue;
    
    clf,subplot(1,2,1);imshow(fullImage);
    gpbFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','.mat'));
    k
    if (~exist(gpbFile,'file'))
        continue;
    end
    
    hold on;
    
    plotBoxes2(locs_r(imageInd,[2 1 4 3]));
    % shift the lip detection...
    topleft_ = locs_r(imageInd,1:2);
    [rows cols ~] = BoxSize(locs_r(imageInd,:));
    lipRectShifted = rows*curLipRect/128 + [topleft_ topleft_];
    [rows2 cols2 ~] = BoxSize(lipRectShifted);
    plotBoxes2(lipRectShifted(:,[2 1 4 3]));
    
    startPts = lipRectShifted(1:2) + rows2*startPoint'/100;
    plot(lipRectShifted(1),lipRectShifted(2),'m*');
    
    plot(startPts(1),startPts(2),'ms');
    ori =  allProps(orientation_,ir(k));
    vec = vec*rows2/100;
    quiver(startPts(1),startPts(2),vec(1),vec(2),0,'g','LineWidth',2,'MaxHeadSize',.5);
    imshow(imresize(curImage,.5,'bilinear'));hold on;
    %
    %     fb = FbMake(2,4,1);
    %     fbResponse = FbApply2d(rgb2gray(curImage),fb,'valid');
    %     size(fbResponse)
    %     size(curImage)
    %     imagesc(sum(abs(fbResponse),3)); colormap jet; axis image; colorbar
    % %
    %     FbReconstruct2d(rgb2gray(curImage),fb,0);
    %     pause;
    %     continue;
    startPoint = allProps(5:6,ir(k))/2;
    endPoint = allProps(7:8,ir(k))/2;
    plot(startPoint(1),startPoint(2),'r+');
    plot(endPoint(1),endPoint(2),'g+');
    
    title(curTitle);
    L = load(gpbFile);
    gPb_orient = double(L.gPb_orient);
    ucmFile = strrep(gpbFile,'.mat','_ucm.mat');
    if (exist(ucmFile,'file'))
        load(ucmFile);
        curUCM = ucm;
    else
        curUCM = contours2ucm(gPb_orient);
    end
    regionsFile = strrep(gpbFile,'.mat','_regions.mat');
    if (exist(regionsFile,'file'))
        fprintf('regions file exists! woohoo!\n');
        load(regionsFile);
    else
        regions  = combine_regions(curUCM,.5);
        regionOvp = regionsOverlap(regions);
    end
    
    %     handsFile = fullfile(conf.handsDir,strrep(currentID,'.jpg','.mat'));
    %     L_hands = load(handsFile);
                
    %     figure,imagesc(curUCM)
    
    T_ovp = .9; % don't show overlapping regions...
    region_sel = suppresRegions(regionOvp, T_ovp); % help! we're being oppressed! :-)
    regions = regions(region_sel);
    regionOvp = regionOvp(region_sel,region_sel);
    %     continue;
    origBoundaries = ucm<=.1;
    segLabels = bwlabel(origBoundaries);
    %     figure,imagesc(segLabels)
    segLabels = imdilate(segLabels,[1 1 0]);
    segLabels = imdilate(segLabels,[1 1 0]');
    
    S = medfilt2(segLabels); % close the small holes...
    
    segLabels(segLabels==0) = S(segLabels==0);
    assert(isempty(find(segLabels==0)));
    %     imagesc(S)
    
    %     imagesc(segLabels==0)
    conf.not_crop = true;
    
    conf.not_crop = false;
    segImage = paintSeg(f2,segLabels);
    startPts2 = startPts+[xmin ymin];
    subplot(1,2,2); imagesc(segImage); axis image;
    hold on;plotBoxes2([ymin xmin ymax xmax],'g');
    plot(startPts2(1),startPts2(2),'ms');
            
    %plotBoxes2(lipRectShifted([2 1 4 3]) + [ymin xmin ymin xmin]);
    
    %     endPT = 4wSegments(L,startPt,normalize_vec(vec),im)
    %     verifySegment(f2,startPts2,vec_ori);
    %
    %     pause;
    %     continue;
    %     startPts2 = round(startPts2);
    
    % % % % %     Z_sel = false(dsize(f2,1:2));
    % % % % %     Z_sel(startPts2(2),startPts2(1)) = 1;
    % % % % %     Z_sel = imdilate(Z_sel,ones(15));
    % % % % %     segInds = unique(segLabels(Z_sel));
    % % % % %
    % % % % %     [y_sel_min,dummy_] = find(Z_sel);
    % % % % %     y_sel_min = min(y_sel_min);
    % % % % %     labelSel = ismember(segLabels,segInds).*segLabels;
    % % % % % %     labelSel = fixLabels(labelSel);
    % % % % %     %     figure,imshow(labelSel,[]);
    % % % % %     %     figure,imagesc(segLabels)
    % % % % %     rprops = regionprops(labelSel,'eccentricity','PixelIdxList','PixelList');
    % % % % %     for iProp = 1:length(rprops)
    % % % % %         if (rprops(iProp).Eccentricity > 0)
    % % % % %             curPts = rprops(iProp).PixelList;
    % % % % %             [y,iy]=  sort(curPts(:,2),'ascend');
    % % % % %         %minPts = single(curPts(iy(1),:));
    % % % % %             if (y < y_sel_min)
    % % % % %                 rprops(iProp).Eccentricity = 0;
    % % % % %             end
    % % % % %         end
    % % % % %     end
    % % % % %
    % % % % %
    % % % % %     [eccen,iEccen] = max([rprops.Eccentricity]);
    % % % % %     Z_final = zeros(size(Z_sel));
    % % % % %     Z_final(rprops(iEccen).PixelIdxList) = 1;
    % % % % %     %     figure,imagesc(Z_final);
    % % % % %     [ii,jj] = find(Z_final);
    % % % % %     [ii,iii] = min(ii);
    % % % % %     startPts2 = [jj(iii),ii];
    % % % % %
    % % % % %
    
    % % % %     quiver(startPts2(1),startPts2(2),vec(1),vec(2),'g','LineWidth',2,'MaxHeadSize',.5);
    % % % %
    % % % %     % the start point may fall on the boundary between two segments. a
    % % % %     % workaround is to caculate which segment contains the most points
    
    
    %     followSegments2(regions,startPts2,vec,f2);
    %     continue;
    z_find = zeros(100);
    z_find(curInds) = 1;
    z_find2 = imresize(z_find,rows2/100);
    %
    %     fb = FbMake(2,5,1);
    %     FR = FbApply2d(z_find2,fb,'same',1);
    %     FR2 = FbApply2d(rgb2gray(II),fb,'same',1);
    %
    %     II = imcrop(f2);
    
    %         figure,imshow(z_find2)
    %
    
    %         vv = conv2(im2double(rgb2gray(f2)),z_find2,'same');
    %
    %
    
    if (curFaceBox(:,conf.consts.FLIP))
        z_find = fliplr(z_find);
    end
    [ii,jj] = find(z_find);
    
    %         figure,imreagesc(z_find)
    %         fipgure,imshow(f2)
    %     hold on; plot(allSegPts(:,1)-1,allSegPts(:,2)-1);
    allSegPts = round(bsxfun(@plus,[xmin ymin]+lipRectShifted(1:2),rows2*[jj,ii]/100));
    allSegPts = unique(allSegPts,'rows');
    allSegInds = unique(sub2ind2(size(segLabels),fliplr(allSegPts)));
    
    % % % %
    % just look at all segments around and take the longest one.
    %
    % "stretch" all labels one pixel top and left to remove pesky
    % boundaries between them
    % %     segmentCandidates = unique(segLabels(allSegInds));
    % %     LL = segLabels.*ismember(segLabels,segmentCandidates);
    % %     nL = renumberregions(LL);
    % %     nLprops = regionprops(nL,'Image','PixelIdxList');
    % %     cN = zeros(size(nLprops));
    % %     for iN = 1:length(nLprops)
    % % %         figure,imshow(nLprops(iN).Image)
    % %           cN(iN) = max(max(normxcorr2(double(nLprops(iN).Image),z_find2)));
    % %     end
    % %     [mm,imm] = max(cN);
    % %     segmentInd = segLabels(nLprops(imm).PixelIdxList(1));
    
    segmentInd = mode(segLabels(allSegInds));
    
    
    %     segmentInd = segLabels(168,425)
    % segmentInd = segLabels(112,222)
    realPts = allSegPts(segLabels(allSegInds)==segmentInd,:);
    
    [m,im] = sort(realPts(:,2),'ascend');
    startPts2 = realPts(im(1),[1 2]);
    strawBoxSize = 40;
    strawBox = [startPts2-floor(strawBoxSize/2),startPts2+ceil(strawBoxSize/2)-1];
    subplot(1,2,2);    plotBoxes2(strawBox([2 1 4 3]),'r','LineWidth',2);
    
    u = unique(cropper(segLabels,strawBox));
    % % %
    % % %     for iu = 1:length(u)
    % % %         [y,x] = find(segLabels==u(iu));
    % % %         [ymin,iymin] = min(y);
    % % %         startPts2 = [x(iymin) ymin];
    % % %         endPt = followSegments(segLabels,startPts2,vec_ori',f2,regions,strawBox,origBoundaries,curUCM)
    % % %         pause(.3);
    % % %     end
    
    [~,name,~] = fileparts(currentID);
    outDir = '/home/amirro/storage/res_s40';
    [~,filename,~] = fileparts(currentID);
    resFileName = fullfile(outDir,[filename '.mat']);
    
    if (exist(resFileName,'file'))
        load(resFileName);
        %         return;
    end
    
    regionConfs = cellfun(@(x) x(region_sel),regionConfs,'UniformOutput',false);
    
    
    
    %     clf;
%     regions = fillRegionGaps(regions);
    %     pause;
    % clf;
    
    [bestIMG,bestScore] = followSegments2(regions,startPts2,G(region_sel,region_sel),regionConfs,f2,faceBoxShifted,regionOvp,binaryFactors);
    
    
    %     clf; subplot(1,2,1); imagesc(f2); axis image;
    %     subplot(1,2,2); imagesc(bestIMG); axis image;title(num2str(bestScore));
    pause;
    continue;
    
    
    curHandBoxes = top_bbs_train{imageInd}(:,[2 1 4 3]);
    curHandBoxes = curHandBoxes(1:min(5,size(curHandBoxes,1)),:);
    plotBoxes2(curHandBoxes ,'y','LineWidth',2');
    cupFile = fullfile('/home/amirro/storage/obj_s40/cups/',strrep(imageIDs{imageInd},'.jpg','.mat'));
    L_cup = load(cupFile);
    curCupBoxes = L_cup.boxes(:,[2 1 4 3 ]);
    curCupBoxes = curCupBoxes(1:min(5,size(curHandBoxes,1)),:);
    plotBoxes2(curCupBoxes ,'m','LineWidth',2');
    %         pause;
    %         continue
    tf = strncmp(currentID,true_ids,length(currentID));
    
    
    %     imshow(f2); hold on; plotBoxes2(action_rois(tf,[2 1 4 3]));
    %     checkDirectionalRois(segLabels,startPts2,f2,origBoundaries,curUCM,regions,action_rois(tf,:));
    
    
    
    %     G = regionAdjacencyGraph(regions);
    
     [feats] = extractFeatures(conf,currentID)
    
    regionsPacked = packRegions(regions);
    imshow(regionsPacked{1},[])
     
    
    %     regionGroups = checkDirectionalRois(segLabels,startPts2,f2,origBoundaries,curUCM,regions);
    
    
    % get the results for the classification of this region
    
    
    continue;
    
    clf;
    for iRegionGroup = 1:length(regionGroups)
        regionSubset = regionGroups(iRegionGroup).regionSubset;
        curRegions = regions(regionSubset);
        curRegionImage = sum(cat(3,curRegions{:}),3);
        %         imshow(curRegionImage,[]);
        %         pause;
        scoreImage = -inf(size(curRegionImage));
        for iRegion = 1:length(regionSubset)
            scoreImage(curRegions{iRegion}) = max(regionConfs{2,2}(regionSubset(iRegion)),...
                scoreImage(curRegions{iRegion}));
        end
        imagesc(scoreImage);colormap jet;
        pause;
    end
    
    
    
    
    
    %     displayRegions(f2,regions);
    %         pause;
    %         continue;
    
    %           clf;imagesc(curUCM); axis image;
    %           pause;
    %imagesc(gPb_orient(:,:,8))
    %imagesc(max(gPb_orient,[],3))
    pause;
    continue;
    confidenceMaps = {};
    
    for iDetector = 1:length(qq_t)
        clocs = qq_t(iDetector).cluster_locs;
        clocs = clocs(clocs(:,11) == imageInd,:);
        confidenceMaps{iDetector} = drawBoxes(f2,clocs,(clocs(:,12)),1);
    end
    confidenceMaps = cat(3,confidenceMaps{:});
    
    clf;subplot(1,2,1),imagesc(f2);axis image;
    subplot(1,2,2); imagesc(max(confidenceMaps,[],3));axis image; title(num2str(max(confidenceMaps(:))));
    
    %     pause;
    %     continue;
    % create some neighborhood around startPT
    
    x0 = endPt(1);
    y0 = endPt(2);
    
    box1 = [x0 - 50,y0-10,x0 + 50,y0+90];
    
    [~,bb1] = imcrop(f2);
    %     bb1 = box1;
    
    [ovp,ints,areas] = boxRegionOverlap(imrect2rect(bb1),regions,dsize(f2,1:2));
    
    %     Z = zeros(dsize(f2,1:2));
    %     Z = drawBoxes(Z,box1,1,1)>0;
    %
    [o,io] = sort(ovp,'descend');
    displayRegions(f2,regions,io,0);
    
    for ii = 1:length(io)
        imshow(f2.*(.1+.9*repmat(regions{(ii)},[1 1 3])));
        pause;
    end
    
    %      break
    bowFile = fullfile(conf.bowDir,strrep(currentID,'.jpg','.mat'));
    load(bowFile,'F','bins');
    feats = struct;
    feats.frames = F;
    feats.binsa = row(bins);
    feats.descrs = [];
    hists = getBOWFeatures(conf,model,{f2},{Z},feats);
    
    hists = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5);
    curImageScore = hists'*w;
    
    %imshow(f2.*(.3+.7*repmat(regions{io(1)},[1 1 3])));
    imshow(f2.*(.5+.5*repmat(Z,[1 1 3])));
    hold on; plotBoxes2(box1([2 1 4 3]));
    title(num2str(curImageScore));
    
    
    %subplot(1,2,1);imagesc(sc(cat(3, II, I), 'prob'));axis image;
    %subplot(1,2,2);
    %imagesc(II); axis image;colorbar
    %pause;
    
    %     for ii = 1:1%length(o);
    %         clf;imshow(f2.*(.3+.7*repmat(regions{io(ii)},[1 1 3])));
    %         hold on;
    %         plot(endPt(1),endPt(2),'Marker','^','MarkerSize',12,'MarkerFaceColor','r');
    %         pause;
    %     end
    
    
    % now classify!
    
    
    
    %     hold on;
    %     plot(endPt(1),endPt(2),'Marker','^','MarkerSize',12,'MarkerFaceColor','r');
    %
    %     pause;
    
end
%%
curScore = [curScore{:}];
% figure,plot(curScore);
mmm = paintRule(mmm,tt);
chosenInds = [chosenInds{:}];

m1 = mImage(mmm);
mmm_rotated = mmm;
for k = 1:length(mmm)
    mmm_rotated{k} = imrotate(maskedImages{k},...
        -double(allProps(orientation_,chosenInds(k))),'bilinear');
end
mImage(mmm_rotated);
m2 = mImage(maskedImages);
% imwrite(mImage(get_full_image([chosenImages{:}])),'/home/amirro/notes/images/straw/sorted_2.tif');
% imwrite(mImage(test_faces_2(t_test)),'/home/amirro/notes/images/straw/test_true.tif');
% imwrite(m1,'/home/amirro/notes/images/straw/sorted.tif');
% imwrite(m2,'/home/amirro/notes/images/straw/segments.tif');
%

