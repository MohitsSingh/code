
% Demo for everything starts here.
initializeStuff;
rmpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab/');
addpath('/home/amirro/code/3rdparty/libsvm-3.17/');
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
addpath('~/code/3rdparty/icp');
%%
addpath('benchmarks');
% 1. find out some statistics about the accuracy of detecting the action
% object vs the extent of the mouth, given the ucm segments
% examineSegmentationPerformance;
dlib_landmark_split;
all_rois = {};
all_labels = {};
%%
params.nodes = nodes;
params.phase = 'training';
posClass = 4;
class_ids = [fra_db.classID];
posClassName = fra_db(find(class_ids==posClass,1,'first')).class;
params.isClass = isClass;
params.learning.negOvp = .1;
params.learning.ovpType = 'overlap';
params.learning.posOvp = .5;
params.learning.max_neg_to_keep = 100;
params.testMode = false;
loggerPath = fullfile('~/code/mircs/logs/',[date '.txt']);
params.logger = log4m.getLogger(loggerPath);
params.logger.setLogLevel(params.logger.ALL);
params.logger.setCommandWindowLevel(params.logger.ALL);

%% prepare images for semantic segmentation.
if 0
    if ~exist('~/storage/misc/images_and_masks_x2.mat','file')
        images = {};
        images_detected_face = {};
        masks = {};
        masks_detected_face = {};
        isTrain = [fra_db.isTrain];
        for t = 1:length(fra_db)
            t
            imgData = fra_db(t);
            I = getImage(conf,imgData);
            [I_sub,~,mouthBox] = getSubImage2(conf,imgData,true,[],2);
            [I_sub2,~,mouthBox2]= getSubImage2(conf,imgData,false,[],2);
            %[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
            [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
            [groundTruth2,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox2);
            images{t} = I_sub;
            images_detected_face{t} = I_sub2;
            masks{t} = groundTruth;
            masks_detected_face{t} = groundTruth2;
        end
        save ~/storage/misc/images_and_masks_x2.mat images images_detected_face masks masks_detected_face fra_db
    end
    %%
    %%
    if ~exist(' ~/storage/misc/images_and_masks_full.mat','file')
        images = {};
        masks = {};
        landmarks = {};
        load fra_db_2015_10_08
        isTrain = [fra_db.isTrain];
        for t = 1:length(fra_db)
            t
            %     if ~isTrain(t),continue,end
            imgData = fra_db(t);
            I = getImage(conf,imgData);
            [I_sub,~,mouthBox,facePoly,I] = getSubImage2(conf,imgData,true);
            %[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
            %[mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,true);
            %         mouthBox = [1 1 size2(I,true)];
            [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,true);
            [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
            if isempty(groundTruth)
                groundTruth = false(size2(I));
            end
            images{t} = im2uint8(I);
            faceMask = poly2mask2(facePoly,size2(I));
            faceMask = faceMask &~groundTruth;
            hands = imgData.hands;
            if (isempty(hands))
                handMask = zeros(size2(I));
            else
                if iscell(hands)
                    hands = cat(1,hands{imgData.handsToKeep});
                end
                handMask = double(box2Region(hands,size2(I),false,true));
            end
            faceMask(handMask==1) = 0;
            handMask(groundTruth) = 0;
            curMask = handMask*3+double(groundTruth)+2*double(faceMask);
            
            %         clf; imagesc2(curMask);
            %         plotPolygons(curLandmarks,'g+');
            %         dpc;
            masks{t} = single(curMask);
            landmarks{t} = curLandmarks;
            
            %         dpc
        end
        %
        save ~/code/mircs/images_and_face_obj_full.mat images masks isTrain landmarks fra_db
    end
    
    %%
    if ~exist('~/storage/misc/images_and_masks_x2_w_hands.mat','file')
        images = {};
        masks = {};
        landmarks = {};
        load fra_db_2015_10_08
        isTrain = [fra_db.isTrain];
        for t = 1:length(fra_db)
            t
            %     if ~isTrain(t),continue,end
            imgData = fra_db(t);
            I = getImage(conf,imgData);
            [I_sub,~,mouthBox,facePoly,I] = getSubImage2(conf,imgData,true,[],2);
            %[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
            %[mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,true);
            %         mouthBox = [1 1 size2(I,true)];
            [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,true);
            [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
            I = cropper(I,mouthBox);
            if isempty(groundTruth) || ~isValid
                groundTruth = false(size2(I));
            end
            facePoly = bsxfun(@minus,facePoly,mouthBox(1:2));
            
            images{t} = im2uint8(I);
            faceMask = poly2mask2(facePoly,size2(I));
            faceMask = faceMask &~groundTruth;
            hands = imgData.hands;
            if (isempty(hands))
                handMask = zeros(size2(I));
            else
                hands = bsxfun(@minus,hands,mouthBox([1 2 1 2]));
                if iscell(hands)
                    hands = cat(1,hands{imgData.handsToKeep});
                end
                handMask = double(box2Region(hands,size2(I),false,true));
            end
            faceMask(handMask==1) = 0;
            handMask(groundTruth) = 0;
            curMask = handMask*3+double(groundTruth)+2*double(faceMask);
            curLandmarks = [bsxfun(@minus,curLandmarks(:,1:2),mouthBox(1:2)) curLandmarks(:,3)];
            clf; imagesc2(curMask);
            plotPolygons(curLandmarks,'g+');
            %         dpc;
            masks{t} = single(curMask);
            landmarks{t} = curLandmarks;
            %         dpc
        end
        %
        save ~/storage/misc/images_and_masks_x2_w_hands.mat images masks isTrain landmarks fra_db
    end
end


%% create images from the detected faces.
if ~exist('~/storage/misc/images_detected.mat','file')    
    images = {};
    images_x2 = {};
    load fra_db_2015_10_08
    isTrain = [fra_db.isTrain];
    for t = 1:length(fra_db)
        t        
        imgData = fra_db(t);
        I = getImage(conf,imgData);
        [I_sub,~,mouthBox,facePoly,I] = getSubImage2(conf,imgData,isTrain(t),[]);
        [I_sub2,~,mouthBox,facePoly,I] = getSubImage2(conf,imgData,isTrain(t),[],2);        
        images{t} = im2uint8(I_sub);
        images_x2{t} = im2uint8(I_sub2);        
    end
    %
    save ~/storage/misc/images_detected.mat images images_x2
end
%%
load ~/storage/misc/images_and_masks.mat
load ~/storage/misc/images_and_masks_res;
load /home/amirro/code/mircs/images_and_face_obj_results.mat
%%
for t = 1:length(fra_db)
    t
    fra_db(t).action_obj = results{t};
    %fra_db(t).action_obj.test = seg_maps_detected_faces{t};
    %fra_db(t).action_obj.test = seg_maps_detected_faces{t};
end

%%

% prepare data: count the number of regions required.
nMasks = {}
for t = 1:20:length(fra_db)
    t
    [candidates,ucm2,isvalid] = getCandidateRegions(conf,fra_db(t),[],true);
    nMasks{end+1} = length(candidates.masks);
end
num2str(mean([nMasks{:}]))
%%
% params.phase
% cur_set = isTrain;

% sampleData1 = collectSamples2(conf, fra_db,cur_set,params);
%%
%%
nSegsPerImage = 2000;
featDimPerSeg = 10000;
bytesPerFeat = 4;
nImages = 1215;
nBytes = nSegsPerImage*featDimPerSeg*bytesPerFeat*nImages;
nBytes/10^9;
%% setup candidate extraction phases
nPhase = 0;
nPhase = nPhase+1;
% %phases = struct('name',{},'featureExtractor',{},'learnedThreshold',{},'getCandidates',{},'classifiers',{});
phases = struct('name',{},'nToKeep',{},'alg_phase',{},'classifiers',{});
phases(nPhase).nToKeep = 3;
phases(nPhase).name = 'actionness';
phases(nPhase).alg_phase = CoarseAlgorithmPhase(conf,params,featureExtractor);
nPhase = nPhase+1;
phases(nPhase).nToKeep = 10;
phases(nPhase).name = 'region relation';
phases(nPhase).alg_phase = InteractionAlgorithmPhase(conf,params,featureExtractor);
nPhase = nPhase+1;
phases(nPhase).nToKeep = 10;
phases(nPhase).name = 'region appearance';
phases(nPhase).alg_phase = FineAlgorithmPhase(conf,params,featureExtractor);
params.phases = phases;
%%
% % % %candidates = phases(iPhase).getCandidates(conf,imgData,prev_candidates);
% trainPhases(1).featureExtractor = boxyFeatureExtractor(
params.posClassName = posClassName;
params.classes = posClass;
params.debug = true;
curClassResults = run_pipeline_3(conf, fra_db(1:1:end), params, featureExtractor);
%%
figure,imshow(masks{1})
figure,imshow(fra_db(1).action_obj.train==2)
%%
pointClouds_gt = {};
scales = zeros(size(fra_db));
goods = [fra_db.isTrain];% & [fra_db.classID]==3;
for t = 1:length(fra_db)
    if goods(t)
        %         t
        if any(masks{t}(:)) && nnz(masks{t}) > 10
            t
            pointClouds_gt{t} = region2Pts(masks{t});
            
            
            scales(t) = size(masks{t},1);
        else
            goods(t) = false;
        end
        %pointClouds_gt{t} = region2Pts(fra_db(t).action_obj.train
    end
end
polys = compressMasks(masks);

% %
% for t = 1:length(images)
%     clf; imagesc2(images{t});
%     plotPolygons(polys{t},'r-');
%     dpc
% end
%
% %%
my_pts = pointClouds_gt(goods);
my_scales = scales(goods);
my_polys = polys(goods);

areas = cellfun3(@(x)size(x,1),my_pts);
areas_n = areas./(my_scales(:).^2);
% for t = 1:length(

%%
for t = 1:length(fra_db)
    if isTrain(t),continue,end
    t
    %     break
    curImg = images_detected_face{t};
    curMask = seg_maps_detected_faces{t};
    %clf; displayRegions(curImg,curMask==2)
    curPts = region2Pts(curMask==2);
    s = size(curMask,1);
    p_fixed = pointCloud([curPts,zeros(size(curPts,1),1)]);
    p_fixed = [curPts,zeros(size(curPts,1),1)]';
    curArea = size(curPts,1)/(s^2);
    areaRatios = curArea./areas_n;
    [r,ir] = sort(areaRatios,'descend');
    ir = ir(1:100);
    %     areaRatios = min(areaRatios,1./areaRatios);
    my_pts1 = my_pts(ir);
    my_scales1 = my_scales(ir);
    my_polys1= my_polys(ir);
    mses = zeros(size(ir))
    ;
    %     my_pts1 = my_pts(areaRatios>.7);
    %     my_scales1 = my_scales(areaRatios>.7);
    %     my_polys1= my_polys(areaRatios>.7);
    %     mses = zeros(size(my_pts1));
    %     clf; imagesc2(curImg); dpc;continue
    %     f = find(goods);
    nIterations = 10;
    % for n = 1:5
    %         my_pts1 = my_pts1(candidate_set);
    nIterations = nIterations+1;
    N = length(my_pts1);
    ts = zeros(3,N);
    rs = zeros(3,3,N);
    errs = zeros(1,N);
    for z = 1:length(my_pts1)
        p = s*my_pts1{z}/my_scales1(z);
        %             p_moving = pointCloud([p,zeros(size(p,1),1)]);
        p_moving = [p,zeros(size(p,1),1)]';
        %         [tform, movingReg, rmse] = pcregrigid(p_moving, p_fixed,'MaxIterations',1,'Extrapolate',false);%, varargin)
        [Ricp Ticp ER t_] = icp(p_fixed,p_moving,nIterations);
        ts(:,z) = Ticp;
        rs(:,:,z) = Ricp;
        %         [Ricp Ticp1 ER t] = icp(p_moving,p_fixed , 3);
        errs(z) = ER(end);
        z
    end
    
    
    %  a foreground measure...
    
    %     [r,ir] = sort(errs,'ascend');
    %     %         ir = ir(1:min(length(ir),10));
    %     my_pts1 = my_pts1(ir);
    %     ts = ts(:,ir);
    %     rs = rs(:,:,ir);
    %     errs = errs(ir);
    % end
    %%
    %     z = zeros(size2(curImg))
    [r,ir] = sort(errs,'ascend');
    zz = zeros(size2(curImg));
    
    for it = 1:10:length(ir)
        k = ir(it);
        %         k = it
        %     plotPolygons(my_pts{ir},'r.')
        Ricp = rs(:,:,k);
        Ticp = ts(:,k);
        E = errs(k);
        p = s*my_pts1{k}/my_scales1(k);
        p_moving = [p,zeros(size(p,1),1)]';
        n = size(p_moving,2);
        Dicp = Ricp * p_moving + repmat(Ticp, 1, n);
        curPoly = s*my_polys1{k}/my_scales1(k);
        curPoly = [curPoly zeros(size(curPoly,1),1)]';
        poly_icp = Ricp * curPoly + repmat(Ticp, 1, size(curPoly,2));
        zz = zz + poly2mask2(poly_icp(1:2,:)',zz);
        %         z
        %         continue
        clf; subplot(1,2,1);
        imagesc2(curImg);
        plotPolygons(curPts,'ro');
        plotPolygons(p,'g+');
        subplot(1,2,2); imagesc2(curImg);
        plotPolygons(curPts,'r.');
        plotPolygons(Dicp(1:2,:)','y.');
        plotPolygons(poly_icp','g-');
        title(num2str(errs(k)));
        %         dpc
        dpc(.01);
        %         if it > 3
        % %             break
        %         end
    end
    
    
    clf; subplot(1,3,1); imagesc2(curImg);
    subplot(1,3,2); imagesc2(sc(cat(3,curMask,curImg),'prob'));
    subplot(1,3,3); imagesc2(sc(cat(3,zz,curImg),'prob'));
    dpc
    
end





% [tform, movingReg, rmse] = pcregrigid(moving, fixed, varargin)


%%


%%
params.cand_mode = 'boxes';
cur_set = f_train;
roi_pos_patches = {};roi_neg_patches = {};
patches = {};
labels = {};
nodes = params.nodes;
for it = 1:5:length(cur_set)
    it
    %     profile on
    t = cur_set(it);
    imgData = fra_db(t);
    [I,I_sub,imageData,hasObj]  = collectImageData(conf,imgData,params);
    %     clf;
    faceMask = imageData.faceMask;
    
    % train face-non face area regions.
    regionSampler.boxOverlap = .5;
    % define bounding box size
    bb_size = round(size2(I_sub)/5);
    regionSampler.boxSize = bb_size;
    s = bb_size(1);
    r = round(regionSampler.sampleOnImageGrid(I_sub));
    r = r(:,1:4);
    [r ,bads] = clip_to_image(r,I_sub);
    dontSampleHere = ~faceMask;
    b_out = sum_boxes(single(dontSampleHere),r);
    [~,~,areas] = BoxSize(r);
    b_obj = sum_boxes(single(imageData.roiMask),r);
    b_out = b_out./areas;
    b_obj = b_obj./areas;
    
    goods = b_out < .5;
    b_out = b_out(goods);
    b_obj = b_obj(goods);
    r = r(goods,:);
    r = [r b_obj];
    pick = nms(r,.5);
    r = r(pick,:);
    b_obj = b_obj(pick);
    patches{end+1} = multiCrop2(I_sub,r);
    labels{end+1} = b_obj;
end
%%
patches = cat(2,patches{:});
labels = cat(1,labels{:})';
%%
patch_descs =  featureExtractor.extractFeaturesMulti(patches);
%showSorted(patches,labels,1000);
% addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
rmpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab/');
addpath('/home/amirro/code/3rdparty/libsvm-3.17/');
%%
lambda = .01;
trainOpts = sprintf('-s 11 -B 1 -e %f',lambda);
my_model = train(double(labels(:)),sparse(double(patch_descs)), trainOpts,'col');

%% test it!

params.cand_mode = 'boxes';
cur_set = f_test;
roi_pos_patches = {};roi_neg_patches = {};
% patches = {};
% labels = {};
nodes = params.nodes;
for it = 51:length(cur_set)
    it
    
    
    %     profile on
    t = cur_set(it);
    imgData = fra_db(t);
    if imgData.classID~=4,continue,end
    [I,I_sub,imageData,hasObj]  = collectImageData(conf,imgData,params);
    %     clf;
    faceMask = imageData.faceMask;
    
    % train face-non face area regions.
    regionSampler.boxOverlap = .5;
    % define bounding box size
    bb_size = round(size2(I_sub)/5);
    regionSampler.boxSize = bb_size;
    s = bb_size(1);
    r = round(regionSampler.sampleOnImageGrid(I_sub));
    r = r(:,1:4);
    [r ,bads] = clip_to_image(r,I_sub);
    dontSampleHere = ~faceMask;
    b_out = sum_boxes(single(dontSampleHere),r);
    [~,~,areas] = BoxSize(r);
    b_obj = sum_boxes(single(imageData.roiMask),r);
    b_out = b_out./areas;
    b_obj = b_obj./areas;
    
    goods = b_out < .5;
    b_out = b_out(goods);
    b_obj = b_obj(goods);
    r = r(goods,:);
    curPatches = multiCrop2(I_sub,r);
    curFeats =  featureExtractor.extractFeaturesMulti(curPatches);
    scores = predict2(curFeats,my_model);
    r = [r scores];
    
    [map,counts] = computeHeatMap(I_sub,r,'max');
    clf; subplot(1,2,1); imagesc2(I_sub);
    subplot(1,2,2); imagesc2(sc(cat(3,map,I_sub,map),'prob_jet'));
    
    dpc;
    %
    %     r = [r b_obj];
    %     pick = nms(r,.5);
    %     r = r(pick,:);
    %     b_obj = b_obj(pick);
    
    %     labels{end+1} = b_obj;
end
