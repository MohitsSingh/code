%% Experiment 0062 - 23/3/2015
%-----------------------------

% 1. Create a facial landmark localization with "good enough" performance
% 2. Refine the location of facial landmarks locally, around the mouth
% 3. create a model to best predict the configuration of the action object

% prepare the dataset:

%drinking_dataset = struct('imageID',{},'faceBox',{},'objects_gt',{},'landmarks',{},'classID',{});
drinking_dataset = {};
baseDir = '/home/amirro/storage/data/drinking_extended';
subDirs = {'straw','bottle'};
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

% add face detections...
load ~/storage/misc/extended_face_dets_1.mat % extended_face_dets

%%

%imgsDir = '/home/amirro/storage/data/drinking_extended/straw';
imgsDir = '/home/amirro/storage/data/drinking_extended/bottle';
d = dir(fullfile(imgsDir,'*.jpg'));
names = {d.name};
paths = cellfun2(@(x)  fullfile(imgsDir,x),names);


addpath('/home/amirro/code/3rdparty/vedaldi_detection/');
load ~/storage/misc/s40_fra_faces_d_new
% mark the facial landmarks on the training images...
%requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
requiredKeypoints = {'MouthLeftCorner','MouthRightCorner'}
dbPath = '/home/amirro/storage/data/drinking_extended/facial_landmarks';
pos_image_data = struct('imageID',paths);
conf.get_full_image = 1;
for t = 1:length(paths)
    t  
    annotateFacialLandmarks(conf,pos_image_data(t),requiredKeypoints,dbPath);
    annoPath = j2m(dbPath,paths(t));
    %         a = load(annoPath)
    %     end
end
addpath('/home/amirro/code/3rdparty/libsvm-3.17/matlab');

% extract facial landmarks from negative samples.
load ~/storage/misc/s40_fra_faces_d_new.mat
fra_db = s40_fra_faces_d
%%
close  all
imgsDir = '/home/amirro/storage/data/drinking_extended/straw';
d = dir(fullfile(imgsDir,'*.jpg'));
names = {d.name};
paths = cellfun2(@(x)  fullfile(imgsDir,x),names);
%% initialize image data: detect facial landmarks and extract
% the locations of mouth corners, etc.

pos_image_data = prepare_facial_landmarks(conf,face_dets,pos_image_data,facialLandmarkData,param);

face_dets = e_face_dets;
gtDir = '/home/amirro/storage/data/drinking_extended/straw/straw_anno';
param.makeSquare = true;
param.anti_rotate = true;
param.cellSize = 4;
param.windowSize = [5 5]*param.cellSize;
param.imgSize = [1.5 1]*40;
param.debugging = false;
param.rotations = -60:10:60;
inflate_factor = 1.5;
param.objToFaceRatio = .3; % ratio of object bounding box relative to face
param.out_img_size =  ceil(inflate_factor*param.windowSize/param.objToFaceRatio);
minScale = 0;
maxScale = 1;
numOctaveSubdivisions = 2 ;
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
negImagePaths = getNonPersonImageList();

tmpDir = '~/tmp';
% get some non-action faces...
sel_neg_landmarks = find([fra_db.isTrain] & [fra_db.classID]~=conf.class_enum.DRINKING & [fra_db.faceScore] > 0);
sel_neg_landmarks = sel_neg_landmarks(1:40:end);
sel_neg_straws = find([fra_db.isTrain] & [fra_db.classID]~=conf.class_enum.DRINKING & [fra_db.faceScore] > 0);
no_straw_faces = {};
for t = 1:400%length(sel_neg2)
    t/length(sel_neg_straws)
    curImgData = fra_db(sel_neg_straws(t)+1);
    I_orig = getImage(conf,curImgData);
    I_face = cropper(I_orig, round(curImgData.faceBox));
    no_straw_faces{end+1} = I_face;
end

% get landmarks on negative faces.
L = load('~/storage/misc/s40_face_dets2.mat');
facialLandmarkData = initFacialLandmarkData();
image_data_neg = prepare_facial_landmarks(conf,L.s40_person_face_dets(sel_neg_landmarks),fra_db(sel_neg_landmarks),facialLandmarkData,param);
negPaths = multiWrite(no_straw_faces,'~/tmp');
%% obtain ground-truth object patches
[goods,gt_mouth_corners] = loadGroundTruthLandmarks(dbPath,pos_image_data);
all_boxes = cat(1,pos_image_data.faceBox);
all_poses = param.poseMap(all_boxes(:,5));
good_pose = goods & abs(all_poses)<=30;
train_sel = find(good_pose);
train_sel = train_sel(1:2:end);
test_sel = setdiff(find(good_pose),train_sel);
[pos_images,polys,angles] = get_pos_samples(conf,pos_image_data,param,gtDir);
% now we have for each face the position of mouth corners and the x,y,theta
% of the straw : normalize w.r.t mouth coordinates
[I_sub_poss,pos_boxes,pos_factors,~] = getSubImages(conf,pos_image_data(train_sel),param);
poly_centers = cellfun2(@mean,polys(train_sel));
% transform to local coordinate system.
poly_centers = cat(1,poly_centers{:});
% transform gt to the local coordinate system so we don't need to do
% anything funny inside the feature extraction function
gt_mouth_corners(train_sel) = shiftAndScale(gt_mouth_corners(train_sel),pos_boxes,pos_factors);
poly_centers = shiftAndScale(poly_centers,pos_boxes,pos_factors);
gt_feats = extract_straw_feats(gt_mouth_corners(train_sel), poly_centers, -col(angles(train_sel)));
polys_train = shiftAndScale(polys(train_sel),pos_boxes,pos_factors);
% visualize the ground truth
% myVisualize(conf,I_sub_poss,gt_mouth_corners(train_sel),polys_train,poly_centers,-angles(train_sel));
%%
% pos_images = [pos_images,flipAll(pos_images)];
%%
param.debugging = false;
no_straw_faces = multiResize(no_straw_faces,param.out_img_size);
patchModels = train_detector(pos_images(train_sel(1:2:end)),no_straw_faces(1:100),param);
imagesc2(vl_hog('render',patchModels));colormap gray
%%
% extract the mouth regions and apply detection on negative image set
r = cat(1,image_data_neg.faceBox);
r = abs(param.poseMap(r(:,5))) <= 30;
image_data_neg_1 = image_data_neg(r);
[I_sub_negs,boxes_neg,factors_neg,mouth_pts_neg_local,mouth_pts_neg] = getSubImages(conf,image_data_neg_1,param);
% mouth_pts_neg = bsxfun(@rdivide,mouth_pts_neg+boxes_neg(:,12),factors);
% negFaceBoxes = cat(1,image_data_neg_1.faceBox);
negFaceScales = boxes_neg(:,3)-boxes_neg(:,1);
% get detection candidates for these images (negative samples)
param.debugging = false;
get_neg_param = param;
get_neg_param.scales = 1;
neg_dets = run_detector(I_sub_negs,patchModels,get_neg_param);
% myVisualize(conf,image_data_neg_1,mouth_pts_neg,[],[]);
get_pos_param = param;
get_pos_param.scales = 1;
get_pos_param.rotations = 0;
% run on positive detetions to get score disributions
pos_dets = run_detector(pos_images(train_sel),patchModels,get_pos_param);
pos_dets = cat(1,pos_dets{:});
% for z = 1:length(neg_dets)  % visualization of negative examples
%     figure(1),clf;imagesc2(I_sub_negs{z});
%     plotPolygons(neg_dets{z}(:,1:2),'r.');
%     plotBoxes(negFaceBoxes(z,:));
%     plotPolygons(mouth_pts_neg_local{z},'g+');
%     pause
% end
neg_feats = {};
for r = 1:length(neg_dets)
    r
    cur_neg_dets = neg_dets{r};
    n = size(cur_neg_dets,1);
    neg_feats{r} = extract_straw_feats(repmat(mouth_pts_neg_local(r),n,1),... 
    cur_neg_dets(:,1:2), cur_neg_dets(:,3));
    neg_feats{r} = [neg_feats{r}, cur_neg_dets(:,4)];
end

neg_feats = cat(1,neg_feats{:});
pos_feats = [gt_feats pos_dets(:,4)];

% finally, train a classifier
C = 1 ;
% neg_feats = neg_feats(1:100:end,:);
numPos = size(pos_feats,1);
numNeg = size(neg_feats,1);
lambda = 1 / (C * (numPos + numNeg)) ;

% Train an SVM model (see Step 2.2)
x = [pos_feats;neg_feats];
y = [ones(1, size(pos_feats,1)) -ones(1, size(neg_feats,1))] ;
%addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

[I_test,boxes_test,factors_test,mouth_pts_test_loca] = getSubImages(conf,pos_image_data(test_sel),param);
x2(I_test);
testFaceScales = boxes_test(:,3)-boxes_test(:,1);
test_param = param;
test_param.debugging = false;
% profile on
test_dets = run_detector(I_test,patchModels,test_param);
% profile viewer
test_feats = {};
for r = 1:length(test_dets)
    r
    cur_test_dets = test_dets{r};
    n = size(cur_test_dets,1);
    test_feats{r} = extract_straw_feats(repmat(mouth_pts_test_loca(r),n,1),... 
    cur_test_dets(:,1:2), cur_test_dets(:,3));
    test_feats{r} = [test_feats{r}, cur_test_dets(:,4)];
end
%%
model = svmtrain(double(y)',(double(x)),'-s 0 -t 2 -c .1');
model.sv_coef'*model.SVs
%vl_svmtrain
%%
for q = 1:length(test_dets)

f =double(test_feats{q});
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
%     plotPolygons(test_dets{q}(iz(1:100),1:2),'g.');
    plotPolygons(winningDet,'r+','LineWidth',4);    
    plotPolygons(mouth_pts_test_loca(q),'ms','LineWidth',3);    
    quiver(winningDet(1),winningDet(2),10*winningFeats(3),10*winningFeats(4),...
    'LineWidth',2);   
    axis image
    title(num2str(z(u)))
    pause
end
end
