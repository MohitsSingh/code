% scripts for fra feature extraction
% define the set of images on which to run
load ~/code/mircs/s40_fra_faces.mat;
s40BaseDir = '/home/amirro/data/Stanford40/JPEGImages';
% 1 run face detectors in the face rois 
outDir = '~/storage/faces_only_baw';
inputData.inputDir = baseDir;
suffix =[];testing = false;
addpath('~/code/utils');
inputData.inputDir = s40BaseDir;
checkIfNeeded = true;
inputData.fileList = struct('name',{s40_fra_faces.imageID});
%%
faceOnly = true;
testing = false;
run_all_2(inputData,outDir,'fra_faces_baw_face_only',testing,suffix,'mcluster03',checkIfNeeded,[]);

faceOnly = true;
testing = false;
load ~/code/mircs/s40_fra_faces_d.mat;
outDir = '~/storage/faces_only_baw_2';
checkIfNeeded = true;
inputData.inputDir = s40BaseDir;
suffix =[];
inputData.fileList = struct('name',{s40_fra_faces_d.imageID});
run_all_2(inputData,outDir,'fra_faces_baw_face_only',testing,suffix,'mcluster01',checkIfNeeded,s40_fra_faces_d);



%% facial landmarks
%% (2) run my facial landmarks on s40
load ~/code/mircs/s40_fra_faces_d.mat
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/faces_only_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = false;
% inputData.fileList = struct('name',{s40_fra_faces_d.imageID});
% inputData.fileList = struct('name','walking_the_dog_167.jpg');
run_all_2(inputData,outDir,'my_facial_landmarks_faces_only',testing,suffix,'mcluster01',checkIfNeeded,[]);

%% face region segmentation
%% (3) s40 face region segmentation
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/faces_only_seg';
inputData.inputDir = baseDir;
suffix =[];testing = true;
checkIfNeeded = false;
inputData.fileList = struct('name','walking_the_dog_167.jpg');

run_all_2(inputData,outDir,'fra_face_seg_face_only',testing,suffix,'mcluster03',checkIfNeeded,[]);


%% (4) s40 object prediction
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_obj_prediction_faces_only';
inputData.inputDir = baseDir;
suffix =[];testing = false;
checkIfNeeded = true;
inputData.fileList = struct('name',{s40_fra_faces.imageID});
run_all_2(inputData,outDir,'fra_obj_pred_face_only',testing,suffix,'mcluster03',checkIfNeeded,[]);

%% pipeline
% clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/face_only_feature_pipeline_all';
suffix =[];testing = false;
checkIfNeeded = true;
moreParams.testMode = true;
moreParams.keepSegments = false;
inputData.inputDir = baseDir;

total_params = defaultPipelineParams(true);
total_params.features.getAppearanceDNN = true;
total_params.features.getHOGShape = true;
total_params.dataDir = '~/storage/face_only_feature_pipeline_all';
total_params.learning.classifierType = 'svm';
total_params.learning.classifierParams.useKerMap = false;


testing = false;
checkIfNeeded = true;
testing  = false;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,total_params);
testing = false;
outDir = '~/storage/face_only_features_lite_2';
run_all_2(inputData,outDir,'fra_features_lite',testing,suffix,'mcluster01',checkIfNeeded);


%% apply classifiers...
% baseDir = '/home/amirro/d


ata/Stanford40/JPEGImages';
outDir = '~/storage/face_only_classification';
suffix =[];testing = false;
checkIfNeeded = true;
testing  = false;
run_all_2(inputData,outDir,'fra_final_classification',testing,suffix,'mcluster03',checkIfNeeded,total_params);


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
checkIfNeeded = true;
moreParams.testMode = true;
moreParams.keepSegments = true;
moreParams.params = defaultPipelineParams(false);
moreParams.params.features.getAttributeFeats = true;
moreParams.params.prevStageDir = '~/storage/stage_1_subsets';
outDir = '~/storage/s40_fra_feature_pipeline_stage_2';

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

run_all_2(inputData,outDir,'fra_features_lite',testing,suffix,'mcluster03',checkIfNeeded,moreParams);


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



%%
outDir = '~/storage/s40_fra_feature_pipeline_partial_dnn_6_7';
load ~/code/mircs/s40_fra_faces_d.mat;
% test_images = {'fixing_a_car_055.jpg'};
inputData.inputDir = baseDir;
suffix =[];
partial_params = defaultPipelineParams(false);
partial_params.features.dnn_net = init_nn_network();
moreParams.params = partial_params;
% sel_train =  col([s40_fra.isTrain]);
test_images = {s40_fra_faces_d.imageID};
inputData.fileList = struct('name',test_images);
testing = false;checkIfNeeded = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster01',checkIfNeeded,moreParams);
