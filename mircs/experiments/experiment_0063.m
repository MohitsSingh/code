%% Experiment 0062 - 23/3/2015
%-----------------------------
%% 1
if ~exist('isInitialized','var')
    %************************************* some initializations
    initpath
    config
    addpath('/home/amirro/code/3rdparty/vedaldi_detection/');
    addpath('/home/amirro/code/my_landmark_localization/');
    addpath('/home/amirro/code/3rdparty/libsvm-3.17/matlab');
    addpath(genpath('/home/amirro/code/3rdparty/svm-struct/'));
    vl_setupnn
    %************************************* define parameters
    param.makeSquare = true;
    param.anti_rotate = true;
    param.cellSize = 4;
    param.windowSize = [5 5]*param.cellSize;
    param.imgSize = [1.5 1]*40;
    param.debugging = false;    
    param.inflate_factor = 1.5;
    param.objToFaceRatio = .3; % ratio of object bounding box relative to face
    param.out_img_size =  ceil(param.inflate_factor*param.windowSize/param.objToFaceRatio);
    minScale = -.5;
    param.ptsAlongLine = 5;

    maxScale = 1;
    numOctaveSubdivisions = 2;
    if (minScale == maxScale)
        scales = minScale;
    else
        scales = 2.^linspace(...
            minScale,...
            maxScale,...
            numOctaveSubdivisions*(maxScale-minScale+1)) ;
    end
    param.poseMap = [90 -90 30 -30 0 0];
    param.scales = scales;
    
    facialLandmarkData = initFacialLandmarkData();
    %************************************* load gt data
    
    %bbLabeler({'straw'},'/home/amirro/storage/data/drinking_extended/straw','/home/amirro/storage/data/drinking_extended/straw/straw_anno');
    bbLabeler({'bottle'},'/home/amirro/storage/data/drinking_extended/bottle','/home/amirro/storage/data/drinking_extended/bottle/bottle_anno');
    % 1. Create a facial landmark localization with "good enough" performance
    % 2. Refine the location of facial landmarks locally, around the mouth
    % 3. create a model to best predict the configuration of the action object
    % some parameters & data initialization.
        
    % prepare the dataset:
                
    baseDir = '/home/amirro/storage/data/drinking_extended';
    subDirs = {'straw','bottle'};
    %gtDir = '/home/amirro/storage/data/drinking_extended/straw/straw_anno';
    obj_gt_baseDir = '/home/amirro/storage/data/drinking_extended/';
    if (exist('~/storage/data/drinking_dataset_w_faces.mat','file'))
        load ~/storage/data/drinking_dataset_w_faces.mat;
    else
        drinking_dataset = {};        
        ids = 1:length(subDirs);
        for iSub = 1:length(subDirs)
            imgsDir = fullfile(baseDir,subDirs{iSub});
            d = dir(fullfile(imgsDir,'*.jpg'));
            curPaths = fullfile(imgsDir,{d.name});
            s = struct('imageID',curPaths,'classID',iSub);
            drinking_dataset{iSub} = s;
        end
        drinking_dataset = cat(2,drinking_dataset{:});
        %save ~/storage/data/drinking_dataset.mat drinking_dataset
        faceDetectionsPath = '~/storage/misc/extended_face_dets_1.mat';
        % add face detections...
        load(faceDetectionsPath); % extended_face_dets
        drinking_dataset = prepare_facial_landmarks(conf,e_face_dets,drinking_dataset,facialLandmarkData,param);
        save ~/storage/data/drinking_dataset_w_faces.mat drinking_dataset
    end
    requiredKeypoints = {'MouthLeftCorner','MouthRightCorner'}
    landmarks_gt_path = '/home/amirro/storage/data/drinking_extended/facial_landmarks';
    conf.get_full_image = 1;
    for t = 1:length(drinking_dataset)
        t
        annotateFacialLandmarks(conf,drinking_dataset(t),requiredKeypoints,landmarks_gt_path);
    end    
    %     regions = cat(1,regions{:});    
    % 2 Negative images.
    load ~/storage/misc/s40_fra_faces_d_new
    fra_db = s40_fra_faces_d
    % mark the facial landmarks on the training images...
    % extract facial landmarks from negative samples.
    %% initialize image data: detect facial landmarks and extract
    % the locations of mouth corners, etc.
    tmpDir = '~/tmp';
    % get some non-action faces...
    sel_neg = find([fra_db.isTrain] & [fra_db.classID]~=conf.class_enum.DRINKING & [fra_db.faceScore] > 1);
    sel_neg = sel_neg(1:10:end);
    sel_neg_1 = sel_neg(1:2:end);
    sel_neg_2 = sel_neg(2:2:end);
    % neg_1_faces = {};
    % for t = 1:length(sel_neg_1)
    %     t/length(sel_neg_1)
    %     curImgData = fra_db(sel_neg_1(t));
    %     I_orig = getImage(conf,curImgData);
    %     I_face = cropper(I_orig, round(curImgData.faceBox));
    %     neg_1_faces{end+1} = I_face;
    % end
    % get landmarks on negative faces.
    L = load('~/storage/misc/s40_face_dets2.mat');
    
    % do training
    % benchmark
    image_data_neg_path = '~/storage/misc/image_data_neg.mat';
    if ~exist(image_data_neg_path,'file')
        image_data_neg = prepare_facial_landmarks(conf,L.s40_person_face_dets(sel_neg),fra_db(sel_neg),facialLandmarkData,param);
        save(image_data_neg_path,'image_data_neg');
    else
        load(image_data_neg_path);
    end
    isInitialized = true;
end


x2(image_data_neg(1).I); plotPolygons(image_data_neg(1).landmarks_local,'r+')

%[mouth_img,mouth_rect,mouth_pts,xy_near_mouth] = get_mouth_img(I_orig,xy_global,curPose,I,resizeFactor,param.imgSize);

negPaths = multiWrite(neg_1_faces,'~/tmp');

u = multiRead(conf,'~/tmp',[],[],100);

%% obtain ground-truth object patches & train models for detection
class_enum.STRAW = 1;
class_enum.BOTTLE = 2;
type_sel = class_enum.STRAW; 
ratios = [.2 .7];
inflate_factors = [1.5 2];
param.inflate_factor = inflate_factors(type_sel);
param.objToFaceRatio = ratios(type_sel);
train_set = false(size(drinking_dataset));
train_set(1:2:end) = true;
test_set = ~train_set;
labels = [drinking_dataset.classID]==type_sel;
% curPosImages = drinking_dataset([drinking_dataset.classID]==type_sel);
[goods,gt_mouth_corners] = loadGroundTruthLandmarks(landmarks_gt_path,drinking_dataset);
all_boxes = cat(1,drinking_dataset.faceBox);
all_poses = param.poseMap(all_boxes(:,5));
good_pose = goods & abs(all_poses)<=30;
% train_sel = find(good_pose);
% train_sel = train_sel(1:2:end);
% test_sel = setdiff(find(good_pose),train_sel);
curGtDir = fullfile(obj_gt_baseDir,subDirs{type_sel},[subDirs{type_sel} '_anno']);
param.cross_val = 5;
param.layers = 19; % fc7 
% [obj_detector,pos_patches,pos_scores] = train_obj_detector(conf,curPosImages(train_sel),image_data_neg,landmarks_gt_path,curGtDir,param);
% say we have an object detector, now use this to detect and score objects
% on the test set
% param.inflate_factor= 1.5;

param.objToFaceRatio = .3;
train_pos = train_set & goods & labels==type_sel;
train_neg = train_set & goods & labels~=type_sel;
[pos_images_g] = {drinking_dataset(train_pos).I};
pos_images_g = [pos_images_g,flipAll(pos_images_g)];
neg_faces = [{image_data_neg(2:2:end).I}, {drinking_dataset(train_neg).I}];
%neg_faces = {drinking_dataset(train_neg).I};
% neg_g = neg_faces(1:2:end);


pos_feats_g = extractDNNFeats(pos_images_g,param.net,param.layers,false);pos_feats_g = pos_feats_g.x;
neg_feats_g = extractDNNFeats(neg_faces,param.net,param.layers,false);neg_feats_g = neg_feats_g.x;
[x_g,y_g] = featsToLabels(pos_feats_g,neg_feats_g);
[w_g b_g] = vl_svmtrain(x_g, y_g, .001);

% restrict training images/polygons to mouth region
% get_pos_samples(conf,curPosImages(train_sel),param,curGtDir);
% toResize =false;
pos_images = get_pos_samples_restricted(conf,drinking_dataset(train_pos),param,curGtDir,toResize,gt_mouth_corners(train_pos));
pos_images = [pos_images,flipAll(pos_images)];
% 
% for z = 1:length(curPosImages)
%     I = imread(curPosImages(train_sel(z)).imageID);
%     clf; imagesc2(curPosImages(z).I);
% %     clf; imagesc2(I);
%         plotPolygons(curPosImages(z).landmarks_local,'r+');
% %     plotPolygons(pos_polys{z},'g-','LineWidth',2);
%     dpc;
% end

% restrict images to area of mouth.
param.net = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/imagenet-vgg-m.mat');

restricted_negs = sample_restricted(drinking_dataset(train_neg),param);
restricted_negs_2 = sample_restricted(image_data_neg(2:2:end),param);
[p1,n1] = ezDetector_prepare2(pos_images,[restricted_negs restricted_negs_2],param);
%[x,y] = featsToLabels(p1,n1(:,1:size(p1,2)));
[x,y] = featsToLabels(p1,n1);
% train_classifier_pegasos
[w b info] = vl_svmtrain(x, y, .001);
%%
figure(1);
% [I_test,boxes_test,factors_test,mouth_pts_test_local] = getSubImages(conf,drinking_dataset(test_set & good_pose),param);
sel_test = test_set & good_pose;
gt_test = labels(sel_test);
scores = zeros(length(drinking_dataset),2);
%%
for t = 1:length(drinking_dataset)
%for t = length(drinking_dataset):-1:1
    if ~sel_test(t)
        continue
    end
    t/length(drinking_dataset)
% for t = 21
%     I = drinking_dataset(t).I; 
%     curMouthPts = 
    %mouth_pts_test_local{t};
    % define a polygon using the mouth points... 
    curMouthPts = drinking_dataset(t).landmarks_local(mouth_corner_inds,:);
    I = drinking_dataset(t).I;
    bb = anchoredBoxSamples(I,curMouthPts,param);    
%     restrictionPolygon = orientedBB(curMouthPts(1,:),curMouthPts(2,:)-curMouthPts(1,:),1);
%     restriction_bb = pts2Box(restrictionPolygon);
    f_g = extractDNNFeats(I,param.net,param.layers,false);f_g = f_g.x;
    prediction_g = w_g'*f_g+b_g;     
    scores(t,1) = prediction_g;        
    detections = anchoredBoxSamples(I,curMouthPts,param);        
    [detections,feats] = ezDetector(w,I,param,'anchorBoxes',detections); 
    scores(t,2) = max(detections(:,5));
    continue
    [z,counts] = computeHeatMap(I,detections,'max');
    clf; subplot(2,1,1);
    imagesc2(I);
    z(isinf(z)) = min(z(~isinf(z)));
    subplot(2,1,2); imagesc2(sc(cat(3,z,I),'prob'))
    %title(num2str([min(detections(:,5)) max(detections(:,5))]));
    title({['global: ' num2str(prediction_g)],['local: ' num2str(max(detections(:,5)))]});
    pause
end
%%
vl_pr(2*gt_test-1,scores(sel_test,1)+scores(sel_test,2)*0.2);
%%
figure(1); % find discriminative feature by elimination, as in zeiler, fergus,
% "visualizing"
for t = 1:length(I_test)
    I = I_test{t};
    detections = occludeAndExtract(I,param,w_g);
    z = visualizeTerm(detections(:,5),boxCenters(detections),size2(I));
    clf; subplot(2,1,1);
    imagesc2(I);
    subplot(2,1,2); imagesc2(sc(cat(3,exp(-z),I),'prob'))
    max(detections(:,5))
    pause
end
%% try a few negative samples for fun
% image_data_neg = prepare_facial_landmarks(conf,L.s40_person_face_dets(sel_neg),fra_db(sel_neg),facialLandmarkData,param);
%%
figure(1);
for t = 2:2:length(image_data_neg)
%for t = length(image_data_neg)-1:-2:2
% for t = 118:-2:2
% for t = 120
    I = image_data_neg(t).I;
    xy_local = image_data_neg(t).landmarks_local;
    mouth_corner_inds = [35 41];
    if (size(xy_local,1)==39)
        clf; imagesc2(I); pause;continue;
    end    
    %detections = mouthToCandidateBoxes(xy_local,I,para   
    curMouthPts = xy_local(mouth_corner_inds,:);
    f_g = extractDNNFeats(I,param.net,param.layers,false);f_g = f_g.x;
    prediction_g = w_g'*f_g+b_g;
    detections = anchoredBoxSamples(I,curMouthPts,param);
    [detections,feats] = ezDetector(w,I,param,'anchorBoxes',detections);
    [z,counts] = computeHeatMap(I,detections,'max');
    clf; subplot(2,1,1);
    imagesc2(I);
    z(isinf(z)) = min(z(~isinf(z)));
    %     subplot(2,1,2); imagesc2(z);
    subplot(2,1,2); imagesc2(sc(cat(3,z,I),'prob'))
    title({['global: ' num2str(prediction_g)],['local: ' num2str(max(detections(:,5)))]});
    pause
end
%% now do it for some positive sample from stanford 40
sel_pos_s40 = find([fra_db.isTrain] & ([fra_db.classID]==conf.class_enum.DRINKING));% & [fra_db.faceScore] > 1);
sel_pos_s40 = sel_pos_s40;
cur_sel = sel_pos_s40(1:40);
image_data_s40 = prepare_facial_landmarks(conf,L.s40_person_face_dets(cur_sel),fra_db(cur_sel),facialLandmarkData,param);
%%
figure(1);
for t = 1:length(image_data_s40)
    I = image_data_s40(t).I;
    xy_local = image_data_s40(t).landmarks_local;
    mouth_corner_inds = [35 41];
    if (size(xy_local,1)==39)
        clf; imagesc2(I); pause;continue;
    end
    curMouthPts = xy_local(mouth_corner_inds,:);
    f_g = extractDNNFeats(I,param.net,param.layers,false);f_g = f_g.x;
    prediction_g = w_g'*f_g+b_g;
    detections = anchoredBoxSamples(I,curMouthPts,param);  
    %[detections,feats] = ezDetector(w,I,myParam,@restrictTop,restriction_bb);
    [detections,feats] = ezDetector(w,I,param,'anchorBoxes',detections);
    %     [detections,feats] = ezDetector(w,I,param);
    [z,counts] = computeHeatMap(I,detections,'max');    
    %[detections,feats] = ezDetector(w,I,param,
    %     [detections,feats] = ezDetector(w,I,param);        
%     [r,ir] = max(detections(:,5));    
%     x2(I); plotBoxes(detections(ir,:));        
%     z = visualizeTerm(detections(:,5),boxCenters(detections),size2(I));
%     z(~poly2mask2(box2Pts(restriction_bb),size2(z))) = -inf;
    clf; subplot(2,1,1);
    imagesc2(I);
    z(isinf(z)) = min(z(~isinf(z)));
    %     subplot(2,1,2); imagesc2(z);
    subplot(2,1,2); imagesc2(sc(cat(3,z,I),'prob'))
    title({['global: ' num2str(prediction_g)],['local: ' num2str(max(detections(:,5)))]});
    pause
end

%%
% now we have for each face the position of mouth corners and the x,y,theta
% of the straw : normalize w.r.t mouth coordinates

[I_sub_poss,pos_boxes,pos_factors,~] = getSubImages(conf,curPosImages(train_sel),param);
poly_centers = cellfun2(@mean,pos_polys);
poly_centers = cat(1,poly_centers{:});
% transform gt to the local coordinate system
gt_mouth_corners(train_sel) = shiftAndScale(gt_mouth_corners(train_sel),pos_boxes,pos_factors);
poly_centers = shiftAndScale(poly_centers,pos_boxes,pos_factors);
gt_feats = extract_straw_feats(gt_mouth_corners(train_sel), poly_centers, -col(pos_angles));
polys_train = shiftAndScale(pos_polys,pos_boxes,pos_factors);
% visualize the ground truth
% % myVisualize(conf,I_sub_poss,gt_mouth_corners(train_sel),polys_train,poly_centers,-pos_angles);

% do this now using structured learning.
my_mouth_corners = gt_mouth_corners(train_sel);
addpath('detect_obj_structured');
% param.rotations = -60:10:60;

param.rotations = 0:30:150;

%%
param.rotations = 0;
%param.rotations = -60:20:60;
param.angle_loss = true;
param.use_mouth_offset = true;
param.useRotations = true;
% param.init_model = obj_detector(:);
param.debugging = false;
param.offset_factor = 10;
% param.cellSize = 4;
% param.windowSize = param.cellSize*[4 8];
param.use_mouth_offset = 1;
param.offset_ker_map = false;
% param.scales = 1;
[model,param] = learn_detect_obj_structured(I_sub_poss,gt_mouth_corners(train_sel),poly_centers,-pos_angles,param);
%%
w = model.w;
w_hog = w(1:param.nHog);
% w_offset = w(param.nHog+1:end);
w_hog = single(reshape(w_hog,param.w_shape));
w_img = vl_hog('render',w_hog);
x2(w_img)

%%
% [I_test,boxes_test,factors_test,mouth_pts_test_local] = getSubImages(conf,curPosImages(test_sel),param);
%
param.rotations = -60:20:60;
% param.rotations = -60;
close all;
my_model = model;
w = model.w;
% w(1:end-6) = 0;
% w(end-5:end) = 0;
param.avoid_out_of_image_dets = true;
my_model.w = w;
for t = 1:length(I_test)
    I = I_test{t};
    %     I = imrotate(I,30,'bilinear','crop');
    mouth_center = mean(mouth_pts_test_local{t});
    x = struct('img',I,'mouth_center',mouth_center);
    [centers,scores,angles] = detect_with_offset(x,my_model,param);
    [q,iq] = max(scores);
    clf;
    subplot(1,2,1); imagesc2(I);
    [b,ib] = sort(scores,'descend');
    sel_ = ib(1:end);
    centers(iq,:)
    plotPolygons(centers(iq(1),:),'g+','LineWidth',2);
    quiver(centers(iq(1),1),centers(iq(1),2),sind(angles(iq(1)))*10, cosd(angles(iq(1)))*10,'LineWidth',2);
    plotPolygons(mouth_pts_test_local(t),'cd','LineWidth',5);
    z = visualizeTerm(scores(sel_),centers(sel_,:),size2(x.img));
    subplot(1,2,2);imagesc2(sc(cat(3,exp(z),I),'prob'));
    
    %     subplot(2,1,1); imagesc2(I);
    %     subplot(2,1,2); imagesc2(exp(z));
    dpc
end
%%
%%

% dummy.
z = zeros(param.windowSize,'single')
z(:,end/2) = 1;
detector = vl_hog(z,param.cellSize);

%%
close all
clc

figure(1)
param.rotations = -60:20:60
for rot = -60:2:60
    % for t = 1:length(I_test);
    %     rot
    z1 = zeros(size(z)*5,'single');
    z1(1:end/4,end/2) = 1;
    z1 = imrotate(z1,rot,'bilinear','crop');
    
    %     z1 =  I_test{t};
    %x2(z1);
    param.avoid_out_of_image_dets = false;
    cur_img_dets = run_detector_nonms(z1,detector,param);
    scores = cur_img_dets(:,4);
    centers = double(cur_img_dets(:,1:2));
    angles = cur_img_dets(:,3);
    [r,ir] = sort(scores,'descend');
    %s = ir(1:100);
    s = ir;
    v = visualizeTerm(scores(s),centers(s,:),size2(z1));
    clf; imagesc2(sc(cat(3,exp(v),z1),'prob'));
    cur_img_dets(ir(1:1),:)
    plotPolygons(centers(ir(1:1),:),'g+');
    drawnow;
    pause
    
    %     pause(.01)
end
%%

neg_feats = getNegativeSamples(conf,image_data_neg_1,obj_detector,param);

get_pos_param = param;
get_pos_param.scales = 1;
get_pos_param.rotations = 0;
pos_feats = [gt_feats pos_scores(:)];

% finally, train a classifier
% neg_feats = neg_feats(1:100:end,:);
numPos = size(pos_feats,1);
numNeg = size(neg_feats,1);
lambda = 1 / (C * (numPos + numNeg)) ;

% Train an SVM model (see Step 2.2)
x = [pos_feats;neg_feats];
y = [ones(1, size(pos_feats,1)) -ones(1, size(neg_feats,1))] ;
%addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

[I_test,boxes_test,factors_test,mouth_pts_test_local] = getSubImages(conf,drinking_dataset(test_sel),param);
x2(I_test);
testFaceScales = boxes_test(:,3)-boxes_test(:,1);
test_param = param;
test_param.debugging = false;
% profile on
test_dets = run_detector(I_test,obj_detector,test_param);
% profile viewer
test_feats = {};
for r = 1:length(test_dets)
    r
    cur_test_dets = test_dets{r};
    n = size(cur_test_dets,1);
    test_feats{r} = extract_straw_feats(repmat(mouth_pts_test_local(r),n,1),...
        cur_test_dets(:,1:2), cur_test_dets(:,3));
    test_feats{r} = [test_feats{r}, cur_test_dets(:,4)];
end
%%
x_ = x;
x_(:,3:4) = 0;
model = svmtrain(double(y)',(double(x)),'-s 0 -t 1 -c .01');
model.sv_coef'*model.SVs
%vl_svmtrain
%%
for q = 1:length(test_dets)
    f = double(test_feats{q});
    % f(:,3) = 1;
    [~, ~, test_scores] = svmpredict(zeros(size(f,1),1), f, model);
    [z,iz] = sort(test_scores,'descend');
    for u = 1%length(iz)
        k = iz(u);
        disp('test feats:')
        disp(f(k,:))
        %     disp('coef:');disp(model.sv_coef'*model.SVs)
        disp('score:')
        disp(z(u))
        clf; %hold on;
        acosd(f(k,5))
        imagesc(I_test{q});
        winningDet = test_dets{q}(k,1:2);
        winningFeats = f(k,:);
        %plotPolygons(test_dets{q}(iz(1:1000),1:2),'g.');
        plotPolygons(test_dets{q}(:,1:2),'g.');
        plotPolygons(winningDet,'r+','LineWidth',4);
        plotPolygons(mouth_pts_test_local(q),'ms','LineWidth',3);
        quiver(winningDet(1),winningDet(2),10*winningFeats(3),10*winningFeats(4),...
            'LineWidth',2);
        axis image
        title(num2str(z(u)))
        pause
    end
end
