%%%%%% Experiment 9 %%%%%%%
% Nov. 12, 2013

% breaking problem into different sub-classes.
% for each face, extract the following features:
% 1. pose (looking left/right/forward)
% 2. location of mouth
% 3. expected face outline
% Then, given these features, break the problem into sub-classes
% The given sub-classes:
% 1. frontal cup
% 2. side cup
% 3. frontal straw
% 4. side straw
% 5. side bottle.
%
% To detect some of the classes, maybe an occlusion cue should be added in as well.
%

echo off;
if (~exist('toStart','var'))
    initpath;
    weightVector = [1 10 10 0*-.01 10 3 1 1 10 1 0];
    
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/seeds');
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');
    addpath('/home/amirro/code/3rdparty/PCA_Saliency_CVPR2013 - v2');
    addpath('/home/amirro/code/3rdparty/guy');
    config;
    load ~/storage/misc/imageData_new; % which image-data to load? the one by zhu, or my face detection + mouth detection?
    
    %imageData = initImageData;
    toStart = 1;
    conf.get_full_image = false;
    imageSet = imageData.train;
    face_comp = [imageSet.faceLandmarks.c]';
    cur_t = imageSet.labels;
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [8 8];
    conf.detection.params.detect_add_flip = 0;
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t);
    facesPath = fullfile('~/mircs/experiments/common/faces_cropped_new.mat');
    load '/home/amirro/mircs/experiments/experiment_0001/sals_new.mat';
    L = load('/home/amirro/mircs/experiments/experiment_0001_improved/exp_result.mat');
    load(facesPath);
    [objectSamples,objectNames] = getGroundTruth(conf,train_ids,train_labels);
    sfe = MaskShapeFeatureExtractor(conf);
    sfe.absoluteFrame = true;
    posemap = [-90:15:90 inf];
    imageNames = {objectSamples.sourceImage};
    objNames = {objectSamples.name};    
    comps = [imageSet.faceLandmarks.c];
    comps(comps==0) = 14;
    T_saliency = L.train_saliency;
    score_fun = @segment_grading;
end
% [feats,labels] = extractRegionFeatures(conf,regionData,trainSet);

featureData = struct('img',{},'mask',{},'ucm','metadata');
count_ = 0;
falses = struct('img',{},'mask',{},'ucm','metadata');
falseChance = 0;
falseCount = 0;
% 
% for k = 1:100
%     I = faces.train_faces{k};
%     I = imresize(I,[128 128],'bilinear');
%     clf;
%     subplot(1,2,1);
%     imagesc(I); axis image;
%     segs = mexSEEDS(im2uint8((I)),0);
%     segImage = paintSeg(I,segs);
%     subplot(1,2,2);
%     imagesc(segImage); axis image;
%     drawnow
% end

%%

%%


%% try to assess accuracy of pose estimation.

% mImage(faces.test_faces(imageData.test.faceScores > t_score & comps == 14 & cur_t));
% t_score = -.8;
% mImage(faces.train_faces(imageData.train.faceScores > t_score & cur_t'));
% 

% montage3(faces.train_faces(imageData.train.faceScores > t_score & (abs(comps-7) <= 3)));
conf.get_full_image = false;
% feats_train = cell(1,length(imageData.train.imageIDs));
% ir = 1:4000;
% ir = 1:5532;
for q =1:length(imageSet.imageIDs)
   
    k = ir(q);
        k
    q
    if (~cur_t(k))
%                         continue;
    end
    
%     gpbFile = fullfile(conf.gpbDir,strrep(imageSet.imageIDs{k},'.jpg','.mat'));
%     if (~exist(gpbFile,'file'))
%         disp('no gpb for this image');
%         continue;
%     end
    
    if (imageSet.faceScores(k) < -.5)
        continue;
    end
%         'aha'
%         if (imageSet.faceScores(k) >= -.5)
%             'oho'
%             continue;
%         end
%     end
%     
    if (~isempty(feats_train{k} ))
%         continue
    end
%     
    
    %     Rs_train{k} =
    %     k = ir(10) % 8,  10 , 11
    % 7: 2335line too straight (not convex), too long
    %k = 128 app
     feats_train{k} =  extractCandidateFeatures5(conf,imageSet,[],k,true,score_fun,q);
end
if (length(feats_train) < length(imageSet.imageIDs))
    feats_train{length(imageSet.imageIDs)} = [];
end


% save ~/mircs/experiments/experiment_0009/feats_train.mat feats_train
%%
% 1: find cups : prominent regions covering the mouth entirely, and
% largely covering the imSubsResizeface, but not too overlapping with the face.
sel_ = 1:length(imageSet.imageIDs);
% sel_ = find(cur_t);

curFaces = faces.train_faces(sel_);
curFeats = feats_train(sel_);
comps_ = comps(sel_);
poses = posemap(comps_);
faceScores = imageSet.faceScores(sel_);
%%
scores = zeros(size(curFeats));
M = -1000;


for q = 1:length(scores)
    r = curFeats{q};
    if (isempty(r))
        scores(q) = M;
        continue;
    end
    s = score_fun(r);
    [scores(q),is_q] = max(s);
end
%scores(~(~isinf(poses) & abs(poses)>=15)) = M;
% scores(abs(poses) > 90) = M;

% scores(isinf(poses)) = M;
% tt = 45;
% scores(abs(poses)>tt) = M;
% ttt = -.6;
scores = scores -faceScores*1;
t0 = faceScores<-.5;
t1 = faceScores<-.6;
t2 = faceScores<-.7;
 
% 
scores(t2)  = scores(t2)-3;
scores(t1 & ~t2)  = scores(t1 & ~t2)-2;
scores(t0 & ~t1) = scores(t0 & ~t1)-1;
% scores(faceScores<-.4) = M;


%scores(faceScores<ttt)-1;
% rrr = (0*T_saliency.stds+T_saliency.means_inside-1*T_saliency.means_outside)';
% scores = scores+.1*(rrr(sel_)');
% scores(isnan(scores)) = M;
%scores(scores < 900) = min(scores(scores~=M));
scores(scores < -900) = min(scores(scores > -900));
% scores = scores-faceScores;
[prec,rec,aps] = calc_aps2(scores',cur_t(sel_));
[r,ir] = sort(scores,'descend');
s = showSorted(curFaces,scores,150);
% imwrite(s,'~/mircs/experiments/experiment_0005/frontal_cups_rule.png');