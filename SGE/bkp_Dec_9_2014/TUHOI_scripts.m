% run face detection and deep net feature extraction on TUHOI images

baseDir = '~/storage/data/TUHOI';
imageNames = dir(fullfile(baseDir,'*.jpg'));
nImages = length(imageNames);
imageNames = {imageNames.name};
outDir = '~/storage/TUHOI_faces_and_features2';
inputData.inputDir = baseDir;
suffix =[];

inputData.fileList = struct('name',imageNames);
checkIfNeeded = true;
testing = true;
run_all_2(inputData,outDir,'run_single_image_parallel',testing,suffix,'mcluster03',checkIfNeeded,baseDir);


%% facial landmarks
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
test_images_zhu = {fra_db.imageID};
inputData.fileList = struct('name',test_images_zhu);
run_all_2(inputData,outDir,'fra_landmarks_parallel',testing,suffix,'mcluster03',true,[]);

%% face region segmentation
%% (3) s40 face region segmentation
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_face_seg';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/s40_fra.mat;
inputData.fileList = struct('name',test_images);
checkIfNeeded = true;
run_all_2(inputData,outDir,'fra_face_seg',testing,suffix,'mcluster03',checkIfNeeded,[]);
%% my facial landmarks
%% (2) run my facial landmarks on s40
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_my_facial_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = true;
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'my_facial_landmarks',testing,suffix,'mcluster03',false,[]);

%% run the face detection on the fra images as well to localize the faces better.
baseDir = '~/storage/data/aflw_cropped_context/'
outDir = '~/storage/aflw_faces_baw';
inputData.inputDir = baseDir;
suffix =[];testing = false;
addpath('~/code/utils');
[paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
inputData.fileList = struct('name',paths);
run_all_2(inputData,outDir,'fra_faces_baw',testing,suffix,'mcluster03');

%% run the face detector on all the stanford 40 images
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_faces_baw';
clear inputData;
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_2(inputData,outDir,'faces_baw',testing,suffix,'mcluster01');

%% (4) s40 object prediction
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_obj_prediction';
inputData.inputDir = baseDir;
suffix =[];testing = false;
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_obj_pred',testing,suffix,'mcluster03',true);

%% trying to run on all s40
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline_all';
load ~/code/mircs/s40_fra.mat;
nImages = length(s40_fra);
% test_images = {'fixing_a_car_055.jpg'};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = false;
moreParams.testMode = true;
moreParams.keepSegments = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded);

%% stage 1-training: extract appearance features from training images after regions have been sorted
test_images_1 = {s40_fra(img_sel_score & [s40_fra.isTrain]').imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images_1);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = false;
moreParams.testMode = true;
moreParams.keepSegments = true;
moreParams.params = stage_params(2);
outDir = '~/storage/s40_fra_feature_pipeline_stage_2';
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);
%% extract the appearance features for selected regions only on train set
test_images = {fra_db([fra_db.isTrain]).imageID};
% test_images = {fra_db.imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = true;
checkIfNeeded = false;
moreParams.testMode = true;
moreParams.keepSegments = true;
moreParams.params = defaultPipelineParams(false);
moreParams.params.features.getAttributeFeats = true;
moreParams.params.prevStageDir = '~/storage/stage_1_subsets';
outDir = '~/storage/s40_fra_feature_pipeline_stage_2';
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);


%% extract just facial features (mouth + face) from s40

clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline_partial_dnn_6_7';
load ~/code/mircs/s40_fra.mat;
% test_images = {'fixing_a_car_055.jpg'};
inputData.inputDir = baseDir;
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = true;
% moreParams.testMode = true;
% moreParams.keepSegments = true;
partial_params = defaultPipelineParams(false);
% total_params.features.getAppearanceDNN = true;
% total_params.features.getHOGShape = true;
partial_params.features.dnn_net = init_nn_network();
moreParams.params = partial_params;
sel_train =  col([s40_fra.isTrain]);

test_images = {s40_fra(img_sel_score).imageID};
inputData.fileList = struct('name',test_images);
testing = false;checkIfNeeded = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);
%%
test_images = {s40_fra(img_sel_score & ~sel_train).imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);

%% extract _all_ features from _all_ s40
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline_all_everything';
load ~/code/mircs/s40_fra.mat;
% test_images = {'fixing_a_car_055.jpg'};
inputData.inputDir = baseDir;

% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = true;
checkIfNeeded = true;
% moreParams.testMode = true;
% moreParams.keepSegments = true;
% % total_params = defaultPipelineParams(true);
% total_params.features.getAppearanceDNN = true;
% total_params.features.getHOGShape = true;
% total_params.features.dnn_net = init_nn_network();
% moreParams.params = total_params;

sel_train =  col([s40_fra.isTrain]);
test_images = {s40_fra(img_sel_score & sel_train).imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);

test_images = {s40_fra(img_sel_score & ~sel_train).imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);


%% run the face detection in quad resolution on all low-scoring faces...
load ~/code/mircs/s40_fra.mat;
nImages = length(s40_fra);
top_face_scores = zeros(nImages,1);
for t = 1:nImages
    top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
end
min_face_score = 0;
img_sel_low_score = (top_face_scores <= min_face_score);img_sel_low_score = img_sel_low_score(:);
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_faces_baw_low';
clear inputData;
%%
inputData.inputDir = baseDir;
suffix =[];testing = false;
inputData.inputDir = baseDir;
test_images = {s40_fra(img_sel_low_score).imageID};
inputData.fileList = struct('name',test_images);
outDir = '~/storage/s40_faces_baw_4x';
% test_images = {fra_db.imageID};
run_all_2(inputData,outDir,'faces_baw',testing,suffix,'mcluster03',true,struct('resizeFactor',4));





