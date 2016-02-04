%% %%%%% Experiment 0028 %%%%%
% 24/2/2014 : train a deformable part model to find straws, cup, bottles,
% in the face area, where the given input images are only those with high
% scoring faces.
if (0)
    % initialization of paths, configurations, etc
    if (~exist('initialized','var'))
        initpath;
        addpath('/home/amirro/code/3rdparty/sliding_segments');
        addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
        config;
        initialized = true;
        %     init_hog_detectors;
        conf.get_full_image = true;
        load ~/storage/misc/imageData_new;
        %         dpmPath = '/home/amirro/code/3rdparty/voc-release5';
        addpath(genpath(dpmPath));
        addpath(genpath('~/code/3rdparty/geom2d'));
        addpath('/home/amirro/code/3rdparty/Gb_Code_Oct2012');
        addpath('/home/amirro/code/3rdparty/rp-master');
        addpath(genpath('/home/amirro/code/3rdparty/seg_transfer'));
        addpath('~/code/3rdparty/logsample');
        addpath('~/code/3rdparty/export_fig');
%         addpath('/home/amirro/code/3rdparty/pose-release-ver1.2/code-basic');
        
        %     false_images_path  = fullfile(conf.cachedir,'false_for_disc_patches.mat');
        %     L = load(false_images_path);
    end
    
    % imgPath = '/home/amirro/storage/data/Stanford40/JPEGImages/drinking_083.jpg';
    % I = imread(imgPath);
    % configParams = LoadConfigFile(configFile);
    % proposals = RP(I, configParams); %[xmin, ymin, xmax, ymax]
    % proposals(:,5) = 1;
    % H = computeHeatMap(I,proposals,'sum');
    % figure,imagesc(H); figure,imagesc(I);
    %
    % % % visPath = '/home/amirro/mircs/experiments/main';
    % % load ~/mircs/experiments/experiment_0017/ferns.mat;
    newImageData = augmentImageData(conf,newImageData);
    % For each image, we can extract
    % 1. occlusion features (segments)
    % 2. HOG features
    % 3. landmarks
    % 4. face mask, pose
    
    %%
    %%
    trainingData = getTrainingPatches(conf,imageData,newImageData,true); % train_ims, rects, face_rects
    extendedData_dir = '~/data/drinking_extended/';
    trainingData_extended = getTrainingPatches_extended(extendedData_dir); % train_ims, rects, face_rects
    trainingData = rmfield(trainingData,'label');
    trainingData = rmfield(trainingData,'face_rect');
    testData = getTrainingPatches(conf,imageData,newImageData,true,'test'); % train_ims, rects, face_rects
    
    nonPersonIDS = row(getNonPersonIds(conf.VOCopts));
    nonPersonIDS = vl_colsubset(nonPersonIDS,300,'Uniform');
    all_trainingData = [trainingData,trainingData_extended];
    
    for k = 1:length(all_trainingData)
        all_trainingData(k).obj_rect = all_trainingData(k).obj_rect(1:4);
    end
    
    conf.features.winsize = [6 6];
    pos_ims = {};
    for k = 1:length(all_trainingData)
        %clf; imagesc(trainingData(k).img); axis image; hold on;
        pos_ims{k} = cropper(all_trainingData(k).img,...
            round(inflatebbox(all_trainingData(k).obj_rect,[1 1],'both',false)));
        %     plotBoxes(trainingData(k).obj_rect,'m');
        %     plotBoxes(trainingData(k).face_rect,'g');
        %     pause;continue;
    end
    
    % pos_ims = {trainingData.img};
    % get the size of pos_images:
    % % % sz = cellfun2(@size2,pos_ims);
    % % % sz = cat(1,sz{:});
    % % % % figure,hist(sz(:,1))
    % % %
    % % % % pos_ims = cellfun2(@(x) imResample(x,[80 80],'bilinear'),pos_ims);
    % % %
    % % %
    % % % %cat(1,AA.imageID)
    % % % %AA = [trainingData_extended.img_data];
    % % % AA = [testData.img_data];
    % % %
    % % % for k = 1:length(AA)
    % % %     fprintf(1,'%d,%s\n',365+k,AA(k).imageID);
    % % % end
    % % %
    % % % % pos_ims = {testData.img};
    % % % %
    % % % S = multiImage(pos_ims,366:366+length(AA)-1);
    % % % imwrite(S,'~/drinking_test.png');
    % % %
    % % % % % % load ~/storage/data/cache/allVladFeats.mat allVladFeats
    allSubIms = cellfun2(@(x)  imResample(x,[80 80],'bilinear'), {newImageData.sub_image});
    % % %
    % % % className = 'bottle';
    % % % [sel_train,sel_test,all_labels] = getImageSubset(conf,className,newImageData);
    % % % % now check some bottle images...
    % % % r = find(sel_test & all_labels==-1);
    % % %
    % % % %load(fullfile(dpmPath,'models','bottle_final_d'));
    % % % % load('models/bottle_d_final');
    model.actionRoiAngle = 30;
    model.lookDirection = 90;
    
    %%
    %% learn dpm models.
    
    usePasAsNegatives = false;
    suff = '';
    if usePasAsNegatives
        suff = '_pas_neg';
    end
    
    subset_configs = struct('name',{},'obj_classes',{},'obj_angle',{},'face_angle',{},'obj_angle_tol',{});
    n = 1;
    subset_configs(n).name = 'bottle_side';
    subset_configs(n).obj_classes = {'bottle'}';
    subset_configs(n).obj_angle = 90; subset_configs(n).face_angle = 90;
    subset_config(n).obj_angle_tol = 30;
    n = n+1;
    subset_configs(n).name = 'cup_glass_front';
    subset_configs(n).obj_classes = {'cup','glass','can'};
    subset_configs(n).obj_angle = 0; subset_configs(n).face_angle = 0;
    n = n+1;
    subset_configs(n).name = 'cup_glass_side';
    subset_configs(n).obj_classes = {'cup','glass','can'};
    subset_configs(n).obj_angle = 45; subset_configs(n).face_angle = 70;
    n = n+1;
    subset_configs(n).name = 'straw_front';
    subset_configs(n).obj_classes = {'straw'};
    subset_configs(n).obj_angle = 0; subset_configs(n).face_angle = 0;
    n = n+1;
    subset_configs(n).name = 'straw_side';
    subset_configs(n).obj_classes = {'straw'};
    subset_configs(n).obj_angle = 45; subset_configs(n).face_angle = 90;
    n = n+1;
    subset_configs(n).name = 'object_90';
    subset_configs(n).obj_classes = {'cup','glass','can','bottle'};
    subset_configs(n).obj_angle = 90; subset_configs(n).face_angle = 90;
    n = n+1;
    subset_configs(n).name = 'object_45';
    subset_configs(n).obj_classes = {'cup','glass','can','bottle'};
    subset_configs(n).obj_angle = 45; subset_configs(n).face_angle = 90;
    n = n+1;
    subset_configs(n).name = 'object_0';
    subset_configs(n).obj_classes = {'cup','glass','can','bottle'};
    subset_configs(n).obj_angle = 0; subset_configs(n).face_angle = 0;
    n = n+1;
    subset_configs(n).name = 'face_side';
    subset_configs(n).obj_classes = {'cup','glass','can','bottle','straw'};
    subset_configs(n).obj_angle = -1; subset_configs(n).face_angle = 90;
    n = n+1;
    subset_configs(n).name = 'face_front';
    subset_configs(n).obj_classes = {'cup','glass','can','bottle','straw'};
    subset_configs(n).obj_angle = -1; subset_configs(n).face_angle = 0;
    
    scores = [newImageData.faceScore];
    train_sel = [newImageData.isTrain] & scores >= -.6;
    allLabels = [newImageData.label];
    train_neg = [train_sel & ~allLabels];
    
    models = {};
    
    basic_classifiers = {};
    for iConfig = 1:length(subset_configs)
        close all;
        m = readDrinkingAnnotationFile('train_data_to_read.csv');
        m = m(1:365); % this is to exclude the test image from training.
        % show cups, looking left.
        modelName = [subset_configs(iConfig).name suff];
        objClasses = subset_configs(iConfig).obj_classes;
        objNames = {m.objType};
        
        found = false(size(objNames));
        for k = 1:length(objNames)
            found(k) = any(find(cellfun(@any,strfind(objClasses,objNames{k}))));
        end
        f = find(found);
        m = m(f);
        curIms = pos_ims(f);
        %     dd = .1;
        %     mImage(cellfun2(@(x) x(ceil(dd*end):floor((1-dd)*end),ceil(dd*end):floor((1-dd)*end),:),curIms));
        %
        % figure,imshow(multiImage(curIms));
        % occlusion = [m.occlusion];
        % S = showSorted(curIms,[m.obj_orientation]);
        %subsel_ = [m.occlusion]==1 & abs([m.obj_orientation]) <= 45;
        % figure,imshow(multiImage(curIms)); title('before');
        orientations = [m.obj_orientation];
        toFlip = orientations < 0;
        yaws = [m.yaw];
        curIms(orientations < 0) = flipAll(curIms(orientations < 0));
        subsel_ = row(true(size(m)));
        if (subset_configs(iConfig).obj_angle ~=-1)
            subsel_ = subsel_ & abs(abs(orientations) - subset_configs(iConfig).obj_angle) <= 30;
        end
        if (subset_configs(iConfig).face_angle ~=-1)
            subsel_ = subsel_ & abs(abs(yaws) - subset_configs(iConfig).face_angle) <= 30;
        end
        sel_ = f(subsel_);
        toFlip = toFlip(subsel_);
        curIms = curIms(subsel_);
        % % % % % %     %%
        % % % % % %     close all;
        % % % % % %     dy = .2;
        % % % % % %     dx = .2;
        % % % % % %     %s = @(y) cellfun2(@(x) x(ceil(d*end):floor((1-d)*end),ceil(d*end):floor((1-d)*end),:),y);
        % % % % % %     s = @(y) cellfun2(@(x) x(ceil(d*end):floor((1-d)*end),ceil(d*end):end,:),y);
        % % % % % %     curIms2 = s(curIms);
        % % % % % %     mImage(curIms2);
        % % % % % %     lm = [newImageData.faceLandmarks];poses = [lm.c];
        % % % % % %     d_side = 1;
        % % % % % %     isSide = poses <= 7-d_side; isSide_flip = poses >= 7+d_side;
        % % % % % %     ims_neg2 = [s(allSubIms(train_neg & isSide)),flipAll(s(allSubIms(train_neg & isSide_flip)))];
        % % % % % %     mImage(ims_neg2);
        % % % % % %     patchSize = 2*[24 32];
        % % % % % %     [pos_feats,neg_feats] = patchesToFeats(curIms2,ims_neg2,patchSize);
        % % % % % %     is_valid = faceScores >= -.5;
        % % % % % %     aa = allSubIms(~isTrain & is_valid & isSide);
        % % % % % %     aa_flip = flipAll(allSubIms(~isTrain & is_valid & isSide_flip));
        % % % % % %     aa = [aa,aa_flip];
        % % % % % %     aa_ = aa;
        % % % % % %     aa = s(aa);
        % % % % % %     test_feats = patchesToFeats(aa,[],patchSize);
        % % % % % %     classifier = train_classifier_pegasos(pos_feats,neg_feats,0);
        % % % % % %     scores_ = classifier.w(1:end-1)'*test_feats;
        % % % % % %
        % % % % % %     showSorted(aa,scores_,inf);
        %
        curRects = cat(1,all_trainingData(sel_).obj_rect);
        
        curImageIDS = {};
        for k = 1:length(sel_)
            curImageIDS{k} = all_trainingData(sel_(k)).img_data.imageID;
        end
        if (usePasAsNegatives)
            trainSet = prepareForDPM(conf,curImageIDS,nonPersonIDS,curRects,toFlip);
        else
            trainSet = prepareForDPM(conf,curImageIDS,allSubIms(train_neg),curRects,toFlip);
        end
        n = 1; % number of subclasses
        valSet = [];
        models{iConfig} = runDPMLearning(modelName, n, trainSet, valSet);
    end
    %save(fullfile(conf.cachedir,'obj_dpm_10_pas_neg.mat'),'models');
    %%
    % now run the object detectors on the new image data...
    curDir = pwd;
    cd ~/code/SGE
    extraInfo.conf = conf;
    extraInfo.models = models;
    extraInfo.path = path;
    %extraInfo.dpmVersion = 4;
    extraInfo.newImageData = newImageData;
    extraInfo.runMode = 'subImage';
    % delete ~/sge_parallel_new/*;
    job_suffix = ['dpm_drink_local' suff];
    justTesting = false;
    outDir = ['~/storage/s40_drink_local' suff];
    mkdir(outDir);
    extraInfo.minFaceScore = -999;
    detections = run_and_collect_results({newImageData.imageID},'detect_dpm_parallel',justTesting,extraInfo,job_suffix,[],outDir);
    detPath = fullfile(outDir,'all.mat');
    load(detPath);
    % save(fullfile(outDir,'all.mat'),'detections');
    
    %%
    load ~/mircs/experiments/experiment_0033/detectors.mat
    detectors{1}.opts.pNms.separate=1;
    
    %%
    % only straw-detectors - center around mouth.
    % = getResponseMap(conf,curImageData,models{iModel},runMode);
    
    for k = 5:length(newImageData)
        k
        curImageData = newImageData(k);
        if (~curImageData.label), continue; end
        %         if (~any(strfind(curImageData.extra.objType,'straw')))
        %             continue;
        %         end
        %         break
        if (curImageData.faceScore < -.6), continue;end
        sub_f = .5;
        %         [I,I_rect] = getImage(conf,curImageData);
        %         clf; imagesc2(I);hold on;
        %         lipBox = curImageData.lipBox+I_rect([1 2 1 2]);
        %         lipBox = inflatebbox(lipBox,[1 2],'both',false);
        %         plotBoxes(lipBox); pause; continue;
        %         M = cropper(I, round(lipBox));
        ff = 1;
        [M,landmarks,I_rect] = getSubImage(conf,curImageData,ff);
        % find the best lip box around...
        ddd = ff*100;
        M = imResample(M,ddd*[1 1],'bilinear');
        curScore = -inf;
        
        for rot = -40:5:40
            M_rot = imrotate(M,rot,'bilinear','crop');
            bbs = acfDetect(M_rot,detectors);
            %     bbs = cat(1,bbs{:});
            if (~isempty(bbs))
                mm = max(bbs(:,5));
                if (mm > curScore)
                    curScore = mm;
                    [b,ib] = sort(bbs(:,5),1,'descend');bbs = bbs(ib(1),:);
                    %                 bbs
                    bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
                    clf; imagesc2(normalise(M_rot));
                    plotBoxes(bbs,'g','LineWidth',2);drawnow; pause(.1);
                end
            end
        end
        continue
        
        clf; imagesc2(M);pause;continue
        M = imResample(M,[80 80],'bilinear');
        [gPb_orient, gPb_thin, textons] = globalPb(M);
        ucm = contours2ucm(double(gPb_orient));
        clf;imagesc(gPb_thin)
        regions  = combine_regions_new(ucm,.1);
        displayRegions(M,regions);
        %         clf;imagesc(M); axis image; hold on; plotBoxes(bb_orig(1,:),'g');
        
        labels = getMultipleSegmentations(im2uint8(M));
        for z = 1:length(labels)
            [segImage,c] = paintSeg(M,labels{z});
            clf; subplot(1,2,1); imagesc2(M);
            subplot(1,2,2); imagesc2(segImage); pause;
            %clf; imagesc(segImage); axis image; drawnow; pause; continue;
        end
        %         title(num2str(max(bb_orig(:,5))));
        %         pause;
    end
    
    %%
    % getAllOcclusionPatterns(conf,newImageData);
    %%
    
    
    objClasses = subset_configs(1).obj_classes;
    labels_ = isTrain & is_valid & labels;
    for k = 1:length(labels_)
        if (labels_(k))
            % labels_(k) = labels_(k) & any(find(cellfun(@any,strfind(objClasses,newImageData(k).extra.objType))));
            %         labels_(k) = labels_(k) & abs(abs(newImageData(k).extra.obj_orientation) - subset_configs(iConfig).obj_angle) <= 40;
            %             if ~strcmp(newImageData(k).extra.objType,classes{iConfig})
            %                 %                 if (abs(newImageData(k).extra.obj_orientation) < 70)
            %                 labels_(k) = false;
            %                 %                 end
            %             end
        end
    end
    
    f = find(labels_);
    addpath('/home/amirro/code/3rdparty/LocalizedActiveContour/');
    segDir = '~/storage/gpb_faces';
    % addpath(genpath('/home/amirro/code/3rdparty/seg_transfer/'));
    n = newImageData(sel_);
    
    
    occlusionScores_bu = occlusionScores;
    %%
    %her:
    % some intereting examples:
    occPath = '/home/amirro/storage/occluders_s40_new4';
    % add occlusion scores to images.
    occlusionScores = -inf(size(newImageData));
    loaded = cell(size(occlusionScores));
    %     loaded_old = loaded;
    %%
    for k = 1:length(newImageData)
        k
        if (test_sel(k))
            %             if (isinf(occlusionScores(k))),continue;end
            curImageData = newImageData(k);
            curImageData.lipScore = lipScores(k);
            f = j2m(occPath, curImageData);
            if (isempty(loaded{k}))
                %                 if (~exist(f,'file'))
                %                     f = j2m('/home/amirro/storage/occluders_s40_new3', curImageData);
                %                 end
                
                L = load(f);
                L.occlusionPattern = rmfield(L.occlusionPattern,'regions');
                loaded{k} = L;
            else
                L = loaded{k};
                if isfield(L.occlusionPattern,'regions')
                    L.occlusionPattern = rmfield(L.occlusionPattern,'regions');
                end
                loaded{k} = L;
                
            end
            
            if (~isempty(L.rprops))
                seg_scores = score_occluders(curImageData,L);
                occlusionScores(k) = max(seg_scores);
                %            break;
            end
            %         end
        end
    end
    
    
    
    
    save ~/storage/misc/loaded_new.mat loaded
    
    %%
    
    %% HERE:
    load ~/mircs/experiments/experiment_0033/detectors.mat
    detectors{1}.opts.pNms.separate = 1;
    
    if (~exist('initialized','var'))
        load;
        initialized = 1;
        addpath(genpath('/home/amirro/code/3rdparty/seg_transfer'));
        initpath;
        config;
        load lipScores;
        load loaded;
    end
end
%%

occPath = '/home/amirro/storage/occluders_s40_new4';
% occPath = '/home/amirro/storage/occluders_s40';
% conf.occlusion.whatFace
% n = newImageData(sel_);
n = newImageData;
% theLipScores = lipScores(sel_);
theLipScores = lipScores;
% 100, 144
% iq = 1:9532;
iq = ir;
for ik = 1:length(iq)
    %     ik = 3 % lady with green bottle
    %     ik = 6
    ik
    k = iq(ik);
    %     k = ik
    curImageData = n(k);
    %      any(find(cellfun(@any,strfind(objClasses
    if (~curImageData.label), continue; end
    if (~any(strfind(curImageData.extra.objType,'straw')))
        continue; end
    curImageData.imageID
    [im,I_rect] = getImage(conf,curImageData.imageID);
    [M,landmarks,face_box,face_poly] = getSubImage(conf,curImageData,.7,true);
    %         [occlusionPatterns,regions,face_mask,mouth_mask] = ...
    %         getOcclusionPattern(conf,curImageData);
    %
    f = j2m(occPath, curImageData);
    if (~exist(f,'file')) disp('doesn''t exist'); continue; end
    L = load(f);
    %     moreFun = @() plotPolygons(face_poly,'g--');
    moreFun = @(varargin) showCoords(face_poly,'color','g',varargin{:});
    curImageData.lipScore = theLipScores(k);
    if (~isempty(L.rprops))
        seg_scores = score_occluders(curImageData,L,im);
        dist_coverage = [L.occlusionPatterns.dist_coverage];
        [u,iu] = sort(seg_scores,'descend');
        figure(1); clf,q = displayRegions(im,L.occludingRegions,seg_scores,0,1,moreFun);
        %figure(1); clf,q = displayRegions(im,L.occludingRegions,seg_scores,0,10,moreFun);
        pause;
    end
    
    %     pause
    
    % show the angular coverage...
end
%%

ik


f = j2m(occPath, curImageData);
L = load(f);
clf; imagesc(im); axis image;
if (~isempty(L.rprops));
    displayRegions(im,L.occludingRegions,[L.rprops.dot]);
end

%%

%     clf; imagesc(im); axis image; pause;
[~,I_rect] = getImage(conf,curImageData.imageID);
[xy_c,mouth_poly,face_poly] = getShiftedLandmarks(curImageData.faceLandmarks,-I_rect);
curImageData.mouth_poly = mouth_poly;
curImageData.face_poly = face_poly;
%     face_mask = poly2mask2(face_poly,size2(im));
%     displayRegions(im,face_mask);
[M,~,face_box,face_poly] = getSubImage(conf,curImageData,2.5,true);
% first, check for agreement between the face polygon and a low-level
% segmentation. Also find regions which may be the actual head region,
%     close all;
segments = {};
for theta = -20:10:200
    theta
    vec = [cosd(theta);sind(theta)];
    sz = size2(M);
    ss = mean(sz);
    %         A = directionalROI(M,sz/2-ss*.1*vec',vec,20);drawnow
    A = directionalROI_rect(M,sz/2-ss*.1*vec',theta,ss/3);
    segments{end+1} = getSegments_graphCut(M,A,128,true);drawnow
    
    % continue
end
break;
segments1 = cellfun2(@(x) imResample(x,sz,'nearest'),segments);
segments1 = shiftRegions(segments1,round(face_box),im);
%         displayRegions(im,segments1);
% continue;
%         figure,imshow(im); hold on;plotBoxes(face_box)
%     displayRegions(M,segments);
%     continue;
%       B =   checkSegmentation(conf,im,face_poly,curImageData);drawnow;continue;
%     I = getImage(conf,curImageData.imageID);
%     [X,Y] = meshgrid(1:size(I,2),1:size(I,1));
%     g_center = (face_box(1:2)+face_box(3:4))/2;
%     g_sigma = mean((face_box(3:4)-face_box(1:2)))/.1;
%     pMap = exp(-((X-g_center(1)).^2+(Y-g_center(2)).^2)/g_sigma);
%     pMap = pMap/max(pMap(:));
%     labelImage = local_segmentation(im,face_box);
%     imshow(im); hold on; plotPolygons(face_poly);plotBoxes(face_box);
%     [regions,~,G] = getRegions(conf,curImageData.imageID,false,segDir);
%     clf;subplot(1,2,2);imagesc(sum(cat(3,regions{:}),3)); axis image;
[M,~,face_box,face_poly] = getSubImage(conf,curImageData,2,false);
%     subplot(1,2,1); imagesc(M); axis image; pause;continue
%     displayRegions(M,regions);
%     clf; imagesc(im); axis image; hold on; plotBoxes(face_box,'g'); pause; continue;
%     face_outline = boxCenters(curImageData.faceLandmarks.xy);
%     res = refineOutline2(conf,curImageData.imageID,face_box,face_poly,true);
%     pause; continue;
face_box = round(face_box);
%     [ucm,gPb_thin] = loadUCM(conf,curImageData.imageID); %#ok<*STOUT>
%     [occlusionPatterns,dataMatrix,regions,~,face_mask,mouth_mask] = ...
%         getOcclusionPattern(conf,curImageData,'toExpand',1,'roi',face_box);
%     imshow(im)
%     [ucm,gPb_thin] = loadUCM(conf,curImageData.imageID); %#ok<*STOUT>
%     imagesc(ucm);axis image
%     figure,imagesc(im);axis image
%         clf,imagesc(im);axis image; pause; continue;
%     curImageData.faceData = getFaceData(curImageData);
%     occPath = j2m(conf.occludersDir,curImageData);

%     load(occPath);
%     curImageData.occlusionPattern = occlusionPattern;
%     imagesc(sum(cat(3,regions{:}),3));
%     displayRegions(im,curImageData.occlusionPattern.regions);
if (~isfield(n(k),'occlusionPattern') || isempty(n(k).occlusionPattern))
    disp('caclulating occlusions...')
    curImageData.occlusionPattern = getOcclusionData(conf,curImageData);
    n(k).occlusionPattern = curImageData.occlusionPattern;
else
    curImageData.occlusionPattern = n(k).occlusionPattern;
end

curImageData.shiftedXY = xy_c;

%     [occludingRegions,occlusionPatterns,rprops] = getOccludingCandidates(im,curImageData);
%     continue;

%     curImageData.curveFeatures = getCurveFeatures(conf,curImageData);
%     displayRegions(im,curImageData.occlusionPattern.regions); title('finished');pause;continue;
%     curImageData.templateResponses = getResponseMap(conf,curImageData,model);
%     H = computeHeatMap(im,curImageData.templateResponses,'max');
%     clf,imagesc(sc(cat(3,H,im),'prob'));max(H(:)) ,pause; continue;
%     displayRegions(im,curImageData.occlusionPattern.regions);
close all;  clf,imagesc(im);axis image;
s = combinedScore(conf,im,curImageData,model);title('finished');pause;continue;
%     clf; imagesc(sc(


% cat(3,s,im),'prob')); axis image; title(num2str(max(s(:))));
pause; continue;



%%
allFisherFeatures = extractFeatures_(fisherFeatureExtractor,allSubIms);
% save -v7.3 ~/storage/data/cache/allFisherFeats.mat allFisherFeatures
load -v7.3 ~/storage/data/cache/allFisherFeats.mat
%%
%classNames = {'bottle','cup','glass','straw'};
classNames = {'bottle','cup','straw'};
models = {};
curIms = pos_ims;
curIms = cellfun2(@(x)  imResample(x,[80 80],'bilinear'), curIms);
[learnParams,conf] = getDefaultLearningParams(conf,1024);
fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
% properties for selection of samples.
lm = [newImageData.faceLandmarks];
sideViews = abs([lm.c]-7)>=2;
trains = [newImageData.isTrain];
pos = [newImageData.label];
validScores = [newImageData.faceScore]>=-.6;

features_pos = extractFeatures_(fisherFeatureExtractor,curIms);
sel_train_neg = trains & ~pos & validScores & sideViews;
negTrainImages = {newImageData(sel_train_neg).sub_image};
features_neg = allFisherFeatures(:,sel_train_neg);

x = [features_pos,features_neg];
y = zeros(size(x,2),1);
y(1:size(features_pos,2)) = 1;
y(size(features_pos,2)+1:end) = -1;
m = mImage(curIms);
classifier = train_classifier_pegasos(features_pos,features_neg,0);

sel_test = ~trains & validScores & sideViews;
mImage({newImageData(sel_test).sub_image});

testImages = {newImageData(sel_test).sub_image};

features_test = allFisherFeatures(:,sel_test);
% % features_test_flip = extractFeatures_(vladFeatureExtractor,flipAll(testImages));
[~,h] = classifier.test(double(features_test));
% [~,h_flip] = classifier.test(double(features_test_flip));

% h = classifier.w(1:end-1)'*features_test;
showSorted(testImages,h,100);


% labels = [newImageData.label];
% labels = labels(~trains);
plot(labels);
sel_ = ~[newImageData.isTrain];
labels_ = [newImageData.label]; labels_ = labels_(sel_);
% finalScores = finalScores(sel_);
finalScores = -inf(size(newImageData));
finalScores(sel_test) = h;
figure,vl_pr(2*labels_-1,finalScores(sel_));
% [r,ir] = sort(h,'descend');
% features_test_flip = extractFeatures_(fisherFeatureExtractor,flipAll(testImages(ir(1:100))));

% [~,h_flip] = classifier.test(double(features_test_flip));
% h_max = h;
% h_max(ir(1:100)) = max(h_max(ir(1:100)),h_flip);
% showSorted(testImages,h_max,100);
features_neg = extractFeatures_(fisherFeatureExtractor,negTrainImages);

%%
load(fullfile(conf.cachedir,'obj_dpm.mat'))
resDir = '~/mircs/experiments/experiment_0028/';
mkdir(resDir);

%%
fff = '/home/amirro/storage/data/drinking_extended/cup/';
cupModelDPM = models{2};
files = dir(fullfile(fff,'*.jpg'));
for k = 5:length(files)
    im = imread(fullfile(fff,files(k).name));
    if (size(im,1) > 640)
        im = imresize(im,[640 NaN],'bilinear');
    end
    [ds, bs] = imgdetect(im, cupModelDPM,cupModelDPM.thresh);
    ds = ds(nms(ds,.5),:);
    showboxes(im,ds(1:min(1,size(ds,1)),:));
    pause
end

% 24/2/2014
%%
for ik = 1:length(sel_)
    k = sel_(ik);
    clf; imagesc(trainingData(k).img); axis image; hold on;
    plotBoxes(trainingData(k).obj_rect,'m');
    plotBoxes(trainingData(k).face_rect,'g');
    pause;continue;
end

features = getFeatures(conf,trainingData(sel_));
% pos_ims = multiCrop(conf,train_imgs,rects,conf.features.winsize*8);
% mm = mImage(pos_ims);
% face_rects = inflatebbox(face_rects,[.8 .8],'both',false);
% pos_faces = multiCrop(conf,train_imgs,face_rects,[80 80]);
% pos_faces = pos_faces(rects(:,11));
% face_poses = estimateFacePose(pos_faces,ferns);
% p = rects(:,end);
% p_est = -face_poses*180/pi;
% mImage(pos_faces); mImage(pos_ims);
% showSorted(pos_faces,p);
% showSorted(pos_faces,p_est);

curIms = [curIms(:);col(flipAll(curIms))];

% profile on
imgSize = conf.features.winsize*8;
%%phase 1: train using appearance only
initialSamples = imageSetFeatures2(conf,curIms,true,imgSize);
w = sqrt(size(initialSamples,1)/42);
conf.features.winsize = [w w];
nSamples = size(initialSamples,2);
[IC,C] = my_kmeans2(initialSamples',1,...
    struct('nTrial',20,'outFrac',0,...
    'display',1,'minCl',3,'metric','sqeuclidean'));
outliers = IC == -1;
fprintf('fraction of outliers: %0.3f\n',nnz(outliers)/length(IC));
maxPerCluster = inf;
[curClusters,ims]= makeClusterImages(curIms',C',IC',initialSamples,[],maxPerCluster);
displayImageSeries(ims);

false_images_path  = fullfile(conf.cachedir,'false_for_disc_patches.mat');
L = load(false_images_path);

clusters = train_circulant(conf,curClusters,L.false_images(1:10:end));

%% phase 2-a : try a nearest neighbor style classification.
%%
%%
Y = [initialSamples,all_neg_feats];
forest = vl_kdtreebuild(Y);
ys = ones(size(Y,2),1);
ys(size(initialSamples,2)+1:end) = 0;


%%
imageSet = imageData.test;
train_ids = imageSet.imageIDs;
train_sel = true(size(train_ids));
train_sel(imageSet.faceScores < -.6) = false;
train_sel(imageSet.labels ==0) = true;
train_sel = row(find(train_sel));
train_sel = vl_colsubset(train_sel,100);
neg_ims = {};
neg_boxes = {};
scale_factors = {};
%%
for ik = 1:length(train_sel)
    %     ik
    k = train_sel(ik)
    %     if (~imageSet.labels(train_sel(ik)))
    %         continue;
    %     end
    [m,~,bbox] = getSubImage(conf,newImageData,train_ids{k});
    [m,scaleFactor] = rescaleImage(m,160,true);
    [X,~,~,~,~,boxes ] = allFeatures( conf,m,1);
    boxes = bsxfun(@plus,boxes(:,1:4)/scaleFactor,bbox([1 2 1 2]));
    % find distance...
    [index,dist] = vl_kdtreequery(forest,Y,single(X),'numneighbors',100,'maxnumcomparisons',100);
    yy = ys(index);
    d = exp(-dist/10);
    d_pos = sum(d.*yy);
    %     d_neg = sum(d.*(1-yy);
    scores = sum(d_pos,1)./sum(d);
    
    %     D_pos = l2(X',initialSamples');
    %     D_neg = l2(X',all_neg_feats');
    %     scores = sum(exp(-D/10),2);
    boxes(:,5) = scores;
    pick =  nms(boxes, .8) ;
    boxes = boxes(pick,:);
    
    
    %     X = vl_homkermap(X,3);
    %     pBoost = struct('verbose',1,'nWeak',128,'pTree',struct('maxDepth',2));
    % pBoost.discrete = 0;
    %     hs = adaBoostApply(X',classifier.model,[],[],1);
    
    %     ik , continue
    % % %     [Yhat f] = classifier.test(X);
    % % %      k
    % % %     boxes(:,5) = f;%classifier.w(1:end-1)'*X;
    I = getImage(conf,train_ids{k});
    H = computeHeatMap(I,boxes,'max');
    %     H = H-min(H(:)); H = H/max(H(:));
    clf; imagesc(sc(cat(3,H,I),'prob')); axis image;
    title(num2str(max(H(:))));
    pause;
    
    %     pause;continue;
    
    neg_ims{ik} = m; neg_boxes{ik} = bbox; scale_factors{ik} = scaleFactor;
end

%% phase 2: collect high-scoring detections from the negative set,
%% as well as positive set, and learn and use both appearance and geometry to train a combined classifier.
toSave = true;
% get some negative images...
imageSet = imageData.train
train_ids = imageSet.imageIDs;
train_sel = true(size(train_ids));
train_sel(imageData.train.faceScores < -.6) = false;
train_sel(imageData.train.labels == 1) = false;
train_sel = row(find(train_sel));
train_sel = vl_colsubset(row(train_sel),500);
ims = {};
boxes = {};
labels = {};
scale_factors = {};

for ik = 1:length(train_sel)
    ik
    k = train_sel(ik);
    [m,~,bbox] = getSubImage(conf,newImageData,train_ids{k});
    [m,scaleFactor] = rescaleImage(m,160,true);
    ims{ik} = m; boxes{ik} = bbox; scale_factors{ik} = scaleFactor;
    labels{ik} = imageSet.labels(k);
end

override = true;
conf.detection.params.detect_save_features = true;
conf.detection.params.detect_max_windows_per_exemplar = 1;
conf.detection.params.detect_add_flip = false;
dets = getDetections(conf,ims,clusters,'','mircs_phase_1',toSave,override);
geom_feats = {};
appearance_feats = {};
for k = 1:length(train_sel)
    k
    cur_bbs = cat(1,dets{k}.bbs{:});cur_bbs = cur_bbs(:,1:4)/scale_factors{k};
    fBox = boxes{k};
    
    cur_bbs = bsxfun(@plus,cur_bbs,fBox([1 2 1 2]));
    
    geom_feats{k} = geometric_features(fBox,cur_bbs)';
    cur_x = cat(2,dets{k}.xs{:});cur_x = cat(2,cur_x{:});
    appearance_feats{k} = cur_x;
    
    % %     I = getImage(conf,imageData.train.imageIDs{train_sel(k)});
    % %     clf; imagesc(I); axis image; hold on;
    % %     plotBoxes(cur_bbs,'g');
    % %     plotBoxes(fBox,'r');
    % %     pause;continue;
    %     all_bbs{k} = cur_bbs;
end
all_neg_feats = [cat(2,geom_feats{:});cat(2,appearance_feats{:})];

pos_feats = {};
for k = 1:length(features)
    pos_feats{k} = [features(k).G(:);features(k).X(:)];
end
pos_feats = cat(2,pos_feats{:});
% classifier = Pegasos(x(:,:),y);
% pos_feats1 = vl_homkermap(pos_feats,3);
% all_neg_feats1 = vl_homkermap(all_neg_feats,3);
% classifier = train_classifier_pegasos(pos_feats1,all_neg_feats1,0);
all_neg_feats = all_neg_feats(6:end,:);
x = [pos_feats(6:end,:),all_neg_feats];
% x = x(6:end,:);
% x = vl_homkermap(x(6:end,:),3);
y = zeros(size(x,2),1);
y(1:size(pos_feats,2)) = 1;
y(size(pos_feats,2)+1:end) = -1;
classifier = Piotr_boosting(double(x),y);

%%
imageSet = imageData.test;
train_ids = imageSet.imageIDs;
train_sel = true(size(train_ids));
train_sel(imageSet.faceScores < -.6) = false;
train_sel(imageSet.labels ==1) = false;
train_sel = row(find(train_sel));
train_sel = vl_colsubset(train_sel,100);
neg_ims = {};
neg_boxes = {};
scale_factors = {};
%%
for ik = 1:length(train_sel)
    %     ik
    k = train_sel(ik)
    [m,~,bbox] = getSubImage(conf,newImageData,train_ids{k});
    [m,scaleFactor] = rescaleImage(m,160,true);
    [X,~,~,~,~,boxes ] = allFeatures( conf,m,1);
    boxes = bsxfun(@plus,boxes(:,1:4)/scaleFactor,bbox([1 2 1 2]));
    %     clf; imagesc(I); axis image; hold on;
    %     plotBoxes(boxes,'g');
    %     plotBoxes(bbox,'r');
    G = geometric_features(bbox,boxes)';
    X = [G;X];
    X = X(6:end,:);
    %     X = vl_homkermap(X,3);
    %     pBoost = struct('verbose',1,'nWeak',128,'pTree',struct('maxDepth',2));
    % pBoost.discrete = 0;
    %     hs = adaBoostApply(X',classifier.model,[],[],1);
    
    %     ik , continue
    [Yhat f] = classifier.test(X);
    k
    boxes(:,5) = f;%classifier.w(1:end-1)'*X;
    I = getImage(conf,train_ids{k});
    H = computeHeatMap(I,boxes,'max');
    %     H = H-min(H(:)); H = H/max(H(:));
    clf; imagesc(sc(cat(3,H,I),'prob')); axis image;
    title(num2str(max(H(:))));
    pause;
    
    %     pause;continue;
    
    neg_ims{ik} = m; neg_boxes{ik} = bbox; scale_factors{ik} = scaleFactor;
end
%%
r = find([newImageData.label]);
r = r(101:end);
% r = r(301:end);
% r = 801:900;
% r = r(5000:end);
% r = 4001:9532;
r = r([newImageData(r).faceScore] > -.4);
imgs = {newImageData(r).sub_image};
% mImage(imgs);
% have for each image

% mImage(ims);
for q = 1:length(r)
    q
    [responses]=detect_in_roi(conf,clusters,newImageData,newImageData(r(q)).imageID,true);
end

r = [newImageData.faceScore] > -.4;
imgs = {newImageData(r).sub_image};
imgs = imgs(cellfun(@(x) ~isempty(x),imgs));
conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_add_flip = 1;
[qq,q] = applyToSet(conf,clusters,imgs,[],'','override',true,'visualizeClusters',true);
[A,AA] = visualizeClusters(conf,imgs,qq,'add_border',...
    false,'nDetsPerCluster',50,'disp_model',false,...
    'disp_model',true,'height',64);
displayImageSeries({A.vis});


[responses]=detect_in_roi(conf,clusters,newImageData,newImageData(r(3)).imageID,true);

mImage(imgs);

r = find([newImageData.label]);r = r(1:100);
w = conf.features.winsize;
s = [newImageData(r).faceScore];
imgs = {newImageData(r).sub_image};
imgs = imgs(s>-.4);
mImage(imgs);

imgs2 = cellfun2(@(x) imresize(x,[128 NaN],'bilinear'),imgs);
% imgs2 = cellfun2(@(x) x(end/3:2*end/3,end/3:3*end/4,:),imgs2);
conf.detection.params.detect_min_scale = .5;

[qq,q] = applyToSet(conf,clusters(4),imgs2,[],'','override',true,'visualizeClusters',true);
[A,AA] = visualizeClusters(conf,imgs2,qq,'add_border',...
    false,'nDetsPerCluster',...
    5,'disp_model',true,'height',64);
displayImageSeries({A.vis});
conf.detection.params.detect_add_flip = 0;

G = fspecial('gauss',size2(ims2{1}),size(ims2{1},1)/4);
G = G/max(G(:));
max_response = zeros(length(imgs2),length(clusters));
for k = 1:length(imgs2)
    k
    im = imgs2{k};
    [rs,boxes]= getResponses(conf,clusters,im);
    bc = round(boxCenters(boxes));
    g_weights = G(sub2ind2(size2(im),fliplr(bc)))';
    for iCluster = 1:length(clusters)
        max_response(k,iCluster) = max(rs(iCluster,:).*g_weights);
    end
end
z = 1;
showSorted(imgs2,max_response(:,z));
[r,ir] = sort(max_response(:,z),'descend');
for k = 1:size(qq.cluster_locs,1)
    k
    t = ir(k)
    im = imgs2{t};
    [rs,boxes]= getResponses(conf,clusters(1),im);
    boxes = clip_to_image(boxes,im);
    bc = round(boxCenters(boxes));
    %     G = fspecial('gauss',size2(im),size(im,1)/10);
    %     G = G/max(G(:));
    boxes(:,5) = rs(:).*G(sub2ind2(size2(im),fliplr(bc)));
    H = computeHeatMap(im,boxes,'max');
    clf,subplot(2,1,1);imagesc(im);axis image;
    subplot(2,1,2);imagesc(H); axis image;
    pause;
end
% H = responsesToHeatMap(rs,boxes);
mImage(imgs2);

% augment image data by all detections, locations of faces, etc.

% first, a mini-demo.
conf.get_full_image = true;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');

ids_ = train_ids;

labels_ = train_labels;
imageSet = imageData.train;
is = 1:length(ids_);
posOnly = true; minFaceScore = -.4;
displayRegionsOneByOne = true;
is = [1142 1131:1132 1143:length(ids_)];
is = 1:length(ids_);

% conf.features.winsize = [8 8];
for ik = 810:length(ids_) % 1
    % for ik = 809
    k = is(ik);
    if (~validImage(imageSet,k,posOnly,minFaceScore))
        continue;
    end
    currentID = ids_{k};
    posemap = 90:-15:-90;
    [M,landmarks,face_box] = getSubImage(conf,newImageData,currentID);
    %     clf; imagesc(M); pause; continue;
    responses = detect_in_roi(conf,clusters,newImageData,currentID,true);
    disp('done')
    pause
    continue
    roi = getActionRoi(M,landmarks);
    [I,I_rect] = getImage(conf,currentID);
    roi = shiftRegions({roi},round(face_box),I);roi = roi{1};
    imageResults = loadImageResults(conf,currentID,newImageData);
    imgData = imageResults.imageData;
    disp('getting occluding regions for current image...');
    [occlusionPatterns,dataMatrix,region_sel,face_mask] = ...
        getOcclusionPattern(conf,imgData,imageResults.regions);
    imageResults.occluders.occlusionPatterns = occlusionPatterns;
    imageResults.occluders.dataMatrix = dataMatrix;
    imageResults.occluders.region_sel = region_sel;
    imageResults.occluders.face_mask = face_mask;
    region_sel = filterOccluders(imageResults.occluders);
    occludingRegions = imageResults.regions(region_sel);
    [ovp,ints,uns] = regionsOverlap(occludingRegions,{roi});
    areas = cellfun(@nnz,occludingRegions(:));
    occludingRegions = occludingRegions(ints./areas > .7);
    
    [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imageSet.faceLandmarks(k),-I_rect);
    
    figure,imagesc(I); hold on; plotPolygons(xy_c,'r.')
    
    visualize some stuff.
    figure(1); clf; vl_tightsubplot(1,2,1); imagesc(I); axis image; hold on;
    if ~isempty(occludingRegions)
        
        if (displayRegionsOneByOne)
            
            plotPolygons(face_poly,'g--','LineWidth',2);
            plotPolygons(mouth_poly,'m','LineWidth',3);
            F = getframe;
            F =  imResample(im2double(F.cdata),size2(I),'bilinear');
            displayRegions(F,occludingRegions,[],0);
        end
    else
        disp('no occluding regions for this image'); pause;
        continue;
    end
    
    if (isempty(occludingRegions))
        
    end
    
    continue;
    
    display the occluding regions
    displayRegions(F,occludingRegions);
    make a union of the occluding regions.
    
    figure(1); clf; vl_tightsubplot(1,2,1); imagesc(I); axis image; hold on;
    Z = zeros(size2(I));
    for q = 1:length(occludingRegions)
        Z = max(Z,q*double(occludingRegions{q}));
        Z = Z+double(occludingRegions{q});
    end
    
    ZZ = sc(cat(3,Z,I),'prob');
    clf,vl_tightsubplot(1,2,1); imagesc(ZZ);axis image; hold on;
    plotPolygons(face_poly,'g--','LineWidth',2);
    plotPolygons(mouth_poly,'m','LineWidth',3);
    
    disp('hit any key to show hands'); pause;
    Z_hands = computeHeatMap(I,imageResults.handLocations(:,[1:4 6]),'max');
    boxes = imageResults.handLocations;
    boxes = boxes(1:min(10,size(boxes,1)),:);
    vl_tightsubplot(1,2,2); imagesc(I); hold on; axis image; plotBoxes(boxes,'g','LineWidth',2);
    saveas(gcf,fullfile(visPath,strrep(currentID,'.jpg','.png')));
    pause;
end


% collect sub images from testing...
%%
M = {};
imageSet = imageData.test;
for k = 1:length(imageSet.imageIDs)
    k
    if (~validImage(imageSet,k,posOnly,minFaceScore))
        continue;
    end
    %     break
    currentID = imageSet.imageIDs{k};
    curIndex = findImageIndex(newImageData,currentID);
    [I,I_rect] = getImage(conf,currentID);
    
    M = getSubImage(conf,newImageData,currentID);
    
    [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(newImageData(curIndex).faceLandmarks,-I_rect);
    mouth_box = pts2Box(mouth_poly);
    face_box = pts2Box(face_poly);
    face_sz = face_box(4)-face_box(2);
    face_box = inflatebbox(mouth_box,face_sz*2,'both',true);
    face_poly = bsxfun(@minus,face_poly,face_box(1:2));
    mouth_poly = bsxfun(@minus,mouth_poly,face_box(1:2));
    newImageData(curIndex).sub_image = cropper(I,round(face_box));
    %     M{end+1} = min(1,max(0,newImageData(curIndex).sub_image));
    %     continue
    %
    clf; imagesc(newImageData(curIndex).sub_image); axis image;
    hold on;
    plotPolygons(face_poly,'g');plotPolygons(mouth_poly,'m');drawnow
    pause
end
%%
% add_suffix = '2';
L = {};

conf.features.winsize = [6 6];
conf.features.padSize = 3;
conf.detection.params.init_params.sbin = 8;
conf.features.winsize = conf.features.winsize + conf.features.padSize;
add_suffix = sprintf('_%d_%d_top_sbin_%d',conf.features.winsize,conf.detection.params.init_params.sbin);
for k = 1:length(classifiers)
    % for k = 2
    k
    detectorPath = fullfile(conf.cachedir,[classifiers(k).class add_suffix '_trained.mat']);
    %     detectorPath = sprintf('~/storage/data/cache/%s_trained.mat',classifiers(k).class);
    if (exist(detectorPath,'file'))
        load(detectorPath); clusters = clusters(:);
        for kk = 1:length(clusters)
            clusters(kk).name = classifiers(k).name;
        end
        L{k}= clusters;
    end
end
L = cat(1,L{:});
clusters = L;
% clusters = rmfield(clusters,'vis');
% clusters = rmfield(clusters,'cluster_samples');
% clusters = rmfield(clusters,'cluster_locs');
%%
save(fullfile(conf.cachedir,'disc_detectors_all.mat'),'clusters');
% curClass = classifiers(9).class;
% load(sprintf('~/storage/data/cache/%s_trained.mat',curClass));
% %  [ Z,Zind,x,y ] = multiImage(M,true,false);
% % MM = M{25}; % 7
% % imshow(MM);
% MM = I;
%%

curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
extraInfo.clusters = clusters;
extraInfo.path = path;
extraInfo.newImageData = newImageData;
job_suffix = 'detect_roi_all';
results = run_and_collect_results(imageData.train.imageIDs,'detect_in_roi_parallel',false,extraInfo,job_suffix,80);
maxScores = zeros(size(results));
mScores = maxScores.*(imageData.train.faceScores > -.6)';

[~,ir] = sort(mScores,'descend');

thetaRange = -60:20:60;

zzz = zeros(121,180);
zzz(26:60,60:120) = 1;
%%

for ik = 1:length(results)
    k = ir(ik)
    r = results{k};
    if (isempty(r.rois)),continue;end
    mm = zeros(size(r.rois));
    for q = 1:length(r.rois)
        m = bsxfun(@times,r.rois{q}.*imresize(zzz,size2(r.rois{q})),r.scoremaps{q});
        maxScores(k) = max(maxScores(k),max(m(:)));
        clf; imagesc(m); axis image; pause;
        mm(q) = max(m(:));
    end
    [z,iz] = max(mm);
    
    clf;
    r.M = imresize(r.M,[180 NaN],'bilinear');
    r.M = min(1,max(0,r.M));
    subplot(1,2,1);imagesc(r.M); axis image;
    I = imrotate(r.M,thetaRange(iz),'bilinear','crop');
    I = I(end/3:end,:,:);
    %
    subplot(1,2,2);imagesc(sc(cat(3,bsxfun(@times,r.rois{iz}.*imresize(zzz,size2(r.rois{1})),...
        r.scoremaps{iz}),I),'prob'));
    axis image;
    pause; continue
end
%%
% detections = cat(1,detections{:});
% imageIndices = cat(1,imageIndices{:});
% [s,is] = sort(imageIndices);
% detections = detections(is);
cd(curDir);

for k = 1:length(imageSet.imageIDs)
    k
    currentID = imageSet.imageIDs{k};
    [responses] = detect_in_roi(conf,clusters,newImageData,currentID)
end


%%
%     %all_scores =

%
%     [r,boxes] = getReponses(conf,clusters,MM_rot);
%     size(r)
%     boxes = boxes(:,1:5);
%     newBoxes = {};
%     for k = 1:size(r,1)
%         boxes(:,5) = r(k,:);
%         newBoxes{k} = boxes;
%     end
%     newBoxes = cat(1,newBoxes{:});
%     newBoxes = clip_to_image(newBoxes,MM_rot);
%     newBoxes(newBoxes(:,5)<0,:) = [];
%     [top,pick] = esvm_nms(newBoxes, .5);
%     D = computeHeatMap(MM_rot,top,'max');
%     figure,imagesc(D);
%     figure,imagesc(MM_rot);
%
%     size(r)
%
%     figure,imagesc(r)
%
%     x = @()applyToSet(conf,clusters,{MM_rot},[],'tmp','override',true,'nDetsPerCluster',10,...
%         'uniqueImages',false,'visualizeClusters',false);
%%

dd = (cat(1,qq.cluster_locs));
figure,plot(dd(:,12))
% apply on the set of data images...
currentClass = classifiers(1).class;
images = dataset_list(dataset,'train',currentClass);
goods = false(size(images));
for k = 1:length(images)
    %         k
    images{k} = sprintf(dataset.VOCopts.imgpath,images{k});
    goods(k) = exist(images{k},'file');
end
images = images(goods);
conf.detection.params.detect_keep_threshold = 0;
conf.detection.params.detect_save_features = 0;
conf.detection.params.detect_max_windows_per_exemplar = 1;
detections = getDetections(conf,images(1:50),clusters,[],[],false);

all_locs = cat(1,clusters_.cluster_locs);
all_c = all_locs(:,6);

ovp = boxesOverlap(all_locs(:,1:4));

classifiers_ovp = zeros(length(clusters_));
for r = 1:size(ovp,1)
    for c = 1:size(ovp,2)
        if (all_c(r) ~= all_c(c))
            if (ovp(r,c) > .7)
                classifiers_ovp(all_c(r),all_c(c)) = classifiers_ovp(all_c(r),all_c(c))+1;
            end
        end
    end
end

imshow(clusters_(27).vis);
figure,imshow(clusters_(7).vis);
imshow(clusters_(46).vis);
figure,imshow(clusters_(37).vis);
%%imagesc(classifiers_ovp); axis image; colorbar


conf.clustering.top_k = inf;
[clusters_] = getTopDetections(conf,detections,clusters,'uniqueImages',true);
clusters_ = removeInvalidClusters(clusters_);
[clusters_1,allImgs] = visualizeClusters(conf,images,clusters_,'height',...
    64,'disp_model',true,'add_border',false,'nDetsPerCluster',inf);
m = clusters2Images(clusters_1);
figure; imagesc(m); axis image;
imwrite(m(1:min(60000,size(m,1)),:,:),'tmp1.jpg');

% % % fullImageDataPath = '~/storage/misc/imageData_full.mat';
% % % if (exist(fullImageDataPath,'file'))
% % %     load(fullImageDataPath);
% % % else
% % %     get image data (faces, names, etc)
% % %     load ~/storage/misc/imageData_new.mat;
% % %     1. detect all occlusions
% % %     newImageData = detect_occluders(conf,newImageData);
% % %     2. detect hands
% % %     newImageData = detect_hands(conf,newImageData);
% % %     3. detect objects
% % %     newImageData = detect_objects(conf,newImageData);
% % % end

%%
%%
% model = initEdgeDetector();
% prepare positive data...
minFaceScore = -.6;
conf.get_full_image = true;
for k = 1:length(newImageData)
    k
    curImageData = newImageData(k);
    if (~curImageData.isTrain) ,continue;end
    if (~validImage(imageSet,k,false,minFaceScore))
        continue;
    end
    currentID = imageSet.imageIDs{k};
    [I,I_rect] = getImage(conf,currentID);
    [M,landmarks,face_box] = getSubImage(conf,curImageData,1,currentID);
    [ucm,E] = loadUCM(conf,currentID);
    E = cropper(E,round(face_box));
    %E = edge(rgb2gray(M),'canny');
    %     M =  imresize(M,2,'bilinear');
    %     [E,Es,O] = edgesDetect( M, model );
    %     [Mag,O] = gradientMag( im2single(M),0,0,0,1 );
    %     clf;  subplot(1,2,1);  imagesc(M); axis image;
    %     subplot(1,2,2); imagesc(Mag); axis image;
    %     pause; continue;
    [regions_,~,G] = getRegions(conf,currentID,false);
    
    
    % correct the face box...
    [occlusionPatterns,dataMatrix,regions,region_sel,face_mask,mouth_mask] = ...
        getOcclusionPattern(conf,newImageData(imageIndex),...
        'regions',regions_,'G',G,'toExpand',1,'roi',round(face_box));
    region_sel = filterOccluders(occlusionPatterns);
    
    %     displayRegions(I,regions_(region_sel));
    % try to fit an ellipse to the edges here.
    %      ellipse_t = fit_ellipse(x,y);
    %     if (isempty(ellipse_t) || length(ellipse_t.status) > 0)
    %         continue;
    %     end
    %     plot_ellipse(ellipse_t);
    %     %
    occludingRegions = regions(region_sel);
    %    displayRegions(I,occludingRegions);
    
    %     edgesToElsd
    %    [ucm,gpb_thin] = loadUCM(conf,currentID);%
    %    ucm = cropper(ucm,round(face_box));
    %    imshow(ucm,[])
    %
    % elsd_gpb
    %     A = edgesToElsd(gpb_thin);
    %     [lines_,ellipses_]=parse_svg(A);
    %     figure,imagesc(I); hold on;plot_svg(lines_,ellipses_);
    
    %     clf; imagesc(M); axis image; pause;continue;
    [roi,roi_out] = getActionRoi(M,landmarks);
    %     clf;imagesc(roi);axis image; pause;continue
    %     roi = shiftRegions({roi},round(face_box),I);roi = roi{1};
    [ovp,ints,uns] = regionsOverlap2(occludingRegions,{roi});
    areas = cellfun(@nnz,occludingRegions(:));
    occludingRegions = occludingRegions(ints./areas > .5);
    occludingRegions = cellfun2(@(x) x & roi_out,occludingRegions);
    occludingRegions = removeDuplicateRegions(occludingRegions);
    
    %      displayRegions(M,occludingRegions);
    %     disp('dfg');
    Z = sum(cat(3,occludingRegions{:}),3);% > 0;
    if (isempty(Z))
        Z = zeros(size2(M));
    end
    %     Z = fillRegionGaps({Z});Z = Z{1};
    %     q = blendRegion(I,Z);
    %     clf; imagesc(q); axis image; hold on; plot(
    %         displayRegions(I,Z);
    %         figure,imagesc(Z)
    clf; imagesc(sc(cat(3,2*Z+face_mask+mouth_mask,M),'prob')); axis image;pause
    
    %figure,imagesc(face_mask+mouth_mask);axis image;
    disp('done');
end

%%
R = poly2mask2(face_poly,size2(im));
%     R = poly2mask2(box2Pts(pts2Box(face_poly)),size2(im));
R = double(cropper(R,round(face_box)));
ss = 128;
M = imresize(M,[ss NaN],'bilinear');
R = imresize(R,[ss NaN],'nearest');

R_false = ~imdilate(R,ones(21));
%
%     imshow(R)
%     [X,Y] = meshgrid(1:size(M,2),1:size(M,1));
%     g_center = fliplr(size2(M))/2;
%     g_sigma = mean(size2(M))/.2;
%     pMap = exp(-((X-g_center(1)).^2+(Y-g_center(2)).^2)/g_sigma);
%
%      pMap = R;
pMap = double(exp(-bwdist(R)/5));

pMap = imfilter(R,fspecial('gaussian',[27 27],6));
pMap(R_false) = 0;
pMap = addBorder(pMap,1,0);
%     pMap = pMap*.8;
%     pMap = pMap/max(pMap(:));
M1 =  vl_xyz2lab(vl_rgb2xyz(im2single(M)));
%     seg = st_segment(M1,pMap,.1,5);
seg = st_segment(im2uint8(M),pMap,.8,5);

segResult = normalise(bsxfun(@times,seg,M));
%     subplot(1,3,1); imagesc(normalise(M)); axis image;
%     subplot(1,3,2); imagesc(pMap); axis image;
%     subplot(1,3,3);imagesc(segResult); axis image;
%
displayRegions(segResult,R);

%clf,imagesc();axis image;
%     drawnow;pause

continue;