% scripts for fra feature extraction
% extract features from all images.
%%
baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feats';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/fra_db.mat;
test_images = {};
%for t = 1:length(fra_db)
test_images = {fra_db.imageID};
%     test_images{t} = fullfile(baseDir,fra_db(t).imageID);
% e%nd
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_extract_feats',testing ,suffix,'mcluster03',true);
%% extract features from predicted locations
% baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
% outDir = '~/storage/s40_fra_pred_feats';
% inputData.inputDir = baseDir;
% suffix =[];testing = false;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db(~[fra_db.isTrain]).imageID};
% inputData.fileList = struct('name',test_images);
% run_all_2(inputData,outDir,'fra_extract_feats_pred',testing ,suffix);
%
% % selective search for the different images...
%
%% run selective search to extract bounding boxes from all fra images
baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_selective_search';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
%test_images = {fra_db([fra_db.isTrain]).imageID};
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_selective_search',testing ,suffix);

%% extract features for all boxes (in the fra regions)....
% % % % % baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
% % % % % outDir = '~/storage/s40_fra_selective_search_feats';
% % % % % inputData.inputDir = baseDir;
% % % % % suffix =[];testing = false;
% % % % % load ~/code/mircs/fra_db.mat;
% % % % % % test_images = {fra_db([fra_db.isTrain]).imageID};
% % % % % test_images = {fra_db.imageID};
% % % % % inputData.fileList = struct('name',test_images);
% % % % % cluster_name = 'mcluster03';
% % % % % run_all_2(inputData,outDir,'fra_selective_search_feats',testing ,suffix,cluster_name);
% % % % %

%% classify all regions (and flip) according to all classifiers
baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_selective_search_classify';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
% test_images = {fra_db([fra_db.isTrain]).imageID};
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
cluster_name = 'mcluster03';
run_all_2(inputData,outDir,'fra_selective_search_classify',testing ,suffix,cluster_name);

%% sum all boxes for all class types
baseDir = '/net/mraid11/export/data/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_selective_search_sum_prob_maps';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
% test_images = {fra_db([fra_db.isTrain]).imageID};
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
cluster_name = 'mcluster03';
run_all_2(inputData,outDir,'fra_selective_search_sum_prob_maps',testing ,suffix,cluster_name);

%% saliency for multiple windows
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_sal_2';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/fra_db.mat;
%test_images = {fra_db([fra_db.isTrain]).imageID};
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'foreground_saliency_multiple_parallel',testing,suffix);

%% facial landmarks
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_landmarks_parallel',testing,suffix,'mcluster01',true,[]);

%% face region segmentation
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_face_seg';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_face_seg',testing,suffix,'mcluster03');

%%
% new face detection, for pose estimation.
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_faces_baw';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_faces_baw',testing,suffix,'mcluster03');

%% run the face detection on the fra images as well to localize the faces better.
baseDir = '~/storage/data/aflw_cropped_context/'
outDir = '~/storage/aflw_faces_baw';
inputData.inputDir = baseDir;
suffix =[];testing = false;
addpath('~/code/utils');
[paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
inputData.fileList = struct('name',paths);
run_all_2(inputData,outDir,'fra_faces_baw',testing,suffix,'mcluster03');

%% extract landmarks using piotr's code (rcpr)
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_keypoints_piotr';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_landmarks_piotr_parallel',testing,suffix,'mcluster01');

%% run the face detector on all the stanford 40 images
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_faces_baw';
clear inputData;
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_2(inputData,outDir,'faces_baw',testing,suffix,'mcluster01');

%% get faces on phrasal recognition dataset
baseDatasetDir = '/home/amirro/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/';
addpath(baseDatasetDir);
baseDir = fullfile(baseDatasetDir,'JPEGImages');
load(fullfile(baseDatasetDir,'groundtruth.mat'));
outDir = '~/storage/VOC3000_faces_baw';
clear inputData;
imageIDs = unique(groundtruth(:,1))

inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
d = dir(baseDir);
% test_images = {fra_db.imageID};
run_all_2(inputData,outDir,'faces_baw',testing,suffix,'mcluster03');

%% facial landmarks on the detected faces of aflw
%% facial landmarks
clear inputData;
baseDir = '~/storage/data/aflw_cropped_context/';
outDir = '~/storage/aflw_zhu_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = false;
run_all_2(inputData,outDir,'fra_landmarks_parallel',testing,suffix,'mcluster03');

%% run the feature extraction pipeline on all of the fra images.
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline';
load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
% test_images = {fra_db([fra_db.isTrain]).imageID};
% test_images = {fra_db(~[fra_db.isTrain]).imageID};
test_images = {fra_db.imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = true;
moreParams.testMode = true;
moreParams.keepSegments = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded);

%% trying to run on all s40
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline_all';
load ~/code/mircs/s40_fra.mat;
nImages = length(s40_fra);
top_face_scores = zeros(nImages,1);
for t = 1:nImages
    top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
end
min_face_score = 0;
img_sel_score = col(top_face_scores > min_face_score);
img_sel_train = col([s40_fra.isTrain]);
img_sel = img_sel_score & ~img_sel_train;
test_images = {s40_fra(img_sel_score).imageID};
% test_images = {'fixing_a_car_055.jpg'};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = true;
moreParams.testMode = true;
moreParams.keepSegments = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded);

%% extract the appearance features for selected regions only on train set
load fra_db;
test_images = {fra_db([fra_db.isTrain]).imageID};
% test_images = {fra_db.imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
% inputData.fileList = struct('name',{ 'smoking_182.jpg'});
suffix =[];testing = false;
checkIfNeeded = true;
moreParams.testMode = true;
moreParams.keepSegments = true;
moreParams.params = defaultPipelineParams(false);
moreParams.params.features.getAttributeFeats = true;
moreParams.params.prevStageDir = '~/storage/stage_1_subsets';
outDir = '~/storage/s40_fra_feature_pipeline_stage_2';
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);

%%
% run again but keep all segments
moreParams.testMode = true;
moreParams.keepSegments = true;
run_all_2(inputData,[outDir '_test'],'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);

% run_all_2(inputData,[outDir '_dummy'],'fra_feature_pipeline',testing || true,suffix,'mcluster03',checkIfNeeded,moreParams);
test_images = {fra_db(~[fra_db.isTrain]).imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
checkIfNeeded = false;
moreParams.testMode = true;
moreParams.keepSegments = true;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,moreParams);

% for t = 1:500
%     t
%     delete(j2m(outDir,fra_db(t)))
% end

%% now for the test image : retain segments.
%test_images = {fra_db.imageID};
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
inputData.inputDir = baseDir;
outDir = '~/storage/fra_feature_pipeline_test_1';
test_images = {fra_db(~[fra_db.isTrain] & [fra_db.classID]~=5).imageID};
inputData.fileList = struct('name',test_images);
suffix =[];testing = false;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03');

%% call cpmc on fra_db as well
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_cpmc';
load ~/code/mircs/fra_db.mat;
%test_images = {fra_db([fra_db.isTrain] & [fra_db.classID]~=5).imageID};
test_images = {fra_db.imageID};
inputData.inputDir = baseDir;
inputData.fileList = struct('name',test_images);
suffix =[];testing = false;
run_all_2(inputData,outDir,'cpmc_parallel2.m',testing,suffix,'mcluster03');


%% run face detection on afw...
baseDir = '/home/amirro/data/afw/testimages';
outDir = '~/storage/afw_faces_baw';
clear inputData;
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/fra_db.mat;
% d = dir(fullfile(baseDir
% test_images = {fra_db.imageID};
run_all_2(inputData,outDir,'faces_baw',testing,suffix,'mcluster03');

%%%%%%%% everything on stanford 40 %%%%%%%%%
%% (1) run face detection on all stanford 40 images -- > finished
% new face detection, for pose estimation.
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/fra_faces_baw';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_faces_baw',testing,suffix,'mcluster03');

%% (2) run my facial landmarks on s40
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_my_facial_landmarks';
inputData.inputDir = baseDir;
suffix =[];testing = false;
% load ~/code/mircs/fra_db.mat;
load ~/code/mircs/s40_fra.mat;
test_images = {s40_fra.imageID};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'my_facial_landmarks',testing,suffix,'mcluster03',false,[]);

%% (3) s40 face region segmentation
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_face_seg';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/s40_fra.mat;
test_images = {s40_fra.imageID};
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
checkIfNeeded = true;
run_all_2(inputData,outDir,'fra_face_seg',testing,suffix,'mcluster03',checkIfNeeded,[]);

%% (4) s40 object prediction
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_obj_prediction';
inputData.inputDir = baseDir;
suffix =[];testing = true;
load ~/code/mircs/s40_fra.mat;
in_orig_fra = false(size(fra_db));
for t = 1:length(fra_db)
    if (~isempty(fra_db(t).indInFraDB) && fra_db(t).indInFraDB~=-1)
        in_orig_fra(t) = true;
    end
end
test_images = {fra_db(~in_orig_fra).imageID};
% test_images = {'fixing_a_car_055.jpg'};
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'fra_obj_pred',testing,suffix,'mcluster03',true);

%% (5) feature extraction
%% run the feature extraction pipeline on alll of the fra images.
clear inputData;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_feature_pipeline';
load ~/code/mircs/s40_fra.mat;
% test_images = {s40_fra.imageID};
test_images = {fra_db.imageID};
inputData.inputDir = baseDir;
% test_images = {'drinking_001.jpg'}
inputData.fileList = struct('name',test_images);
suffix =[];testing = false;
checkIfNeeded = false;
run_all_2(inputData,outDir,'fra_feature_pipeline',testing,suffix,'mcluster03',checkIfNeeded,'aaa');

%% test mode...
test_images = {fra_db([fra_db.isTrain]).imageID};
inputData.fileList = struct('name',test_images);
forceTestMode = true;
testing = true;
outDir = '~/storage/s40_fra_feature_pipeline_test';
run_all_2(inputData,outDir,'fra_feature_pipeline',testing ,suffix,'mcluster03',true,forceTestMode)

%% some corrections
% targetDir = '~/storage/s40_my_facial_landmarks';
% for t = 1:length(fra_db)
%     t
%     delete(fullfile(targetDir,strrep(fra_db(t).imageID,'.jpg','*')))
% end



targetDir = '~/storage/s40_my_facial_landmarks';
for t = 1:length(fra_db)
    t
    delete(j2m(targetDir,fra_db(t)));
end



% % targetDir = '~/storage/s40_fra_face_seg';
% % for t = 1:length(fra_db)
% %     t
% %     delete(fullfile(targetDir,strrep(fra_db(t).imageID,'.jpg','*')))
% % end
% % targetDir = '~/storage/s40_obj_prediction';
% % for t = 1:length(fra_db)
% %     t
% %     delete(fullfile(targetDir,strrep(fra_db(t).imageID,'.jpg','*')))
% % end


% targetDir = '~/storage/s40_fra_feature_pipeline';
% for t = 1:length(fra_db)
%     t
%     delete(fullfile(targetDir,strrep(fra_db(t).imageID,'.jpg','*')))
% end

%%

baseDir = '/home/amirro/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages';
cacheTag = '~/storage/phrasal_rec';
funs = struct('tag',{},'fun',{});
funs(1).tag = 'faces';
funs(1).fun = 'fra_faces_baw_new';
funs(2).tag = 'face_landmarks';
funs(2).fun = 'my_facial_landmarks_new';
funs(3).tag = 'seg';
funs(3).fun = 'fra_face_seg_new';
funs(4).tag = 'obj_pred';
funs(4).fun = 'fra_obj_pred_new';
funs(5).tag = 'feat_pipeline';
funs(5).fun = 'fra_feature_pipeline_new';
funs(6).tag = 'ELSD';
funs(6).fun = 'run_elsd';
funs(7).tag = 'drfi_saliency';
funs(7).fun = 'run_drfi_saliency';
% run a

for t = 1:length(funs)
    funs(t).outDir = fullfile(cacheTag,funs(t).tag);
end
pipelineStruct = struct('baseDir',baseDir,'funs',funs);

%%
params = defaultPipelineParams(true);
params.externalDB = true;
clear inputData;
inputData.inputDir = baseDir;
params.externalDB = true;

% create the phrasal rec db using the face detection+ provided groundtruth

suffix =[];testing = true;
t = 5;checkIfNeeded = false;
run_all_2(inputData,funs(t).outDir,funs(t).fun,testing,suffix,'mcluster03',checkIfNeeded,...
    struct('forceTestMode',false,'pipelineParams',funs,'params',params));
% for t = 5
%     inputData.inputDir = baseDir;
%     suffix =[];testing = true;
%     run_all_2(inputData,funs(t).outDir,funs(t).fun,testing,suffix,'mcluster01',checkIfNeeded,pipelineStruct);
% end

inputData.inputDir = baseDir;
suffix =[];testing = false;
t = 6;
checkIfNeeded = false;
run_all_2(inputData,funs(t).outDir,funs(t).fun,testing,suffix,'mcluster01',checkIfNeeded,...
    struct('forceTestMode',true,'pipelineParams',funs));

inputData.inputDir = baseDir;
suffix =[];testing = true;
t = 7;
checkIfNeeded = false;
run_all_2(inputData,funs(t).outDir,funs(t).fun,testing,suffix,'mcluster01',checkIfNeeded,...
    struct('forceTestMode',true,'pipelineParams',funs));


% sel_ = [s40_fra.isTrain] & [s40_fra.classID]==conf.class_enum.DRINKING & top_face_scores' < -.3;
% displayImageSeries(conf,{s40_fra(sel_).imageID});


%% extract global/rectangle DNN features from all images
%% test mode...
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_global_dnn_feats_m_2048';
inputData.inputDir = baseDir;
load ~/code/mircs/s40_fra
test_images = {s40_fra.imageID};
inputData.fileList = struct('name',test_images);
testing = false;
checkIfNeeded = false;
run_all_2(inputData,outDir,'global_dnn_feats',testing ,[],'mcluster03',checkIfNeeded,[]);

%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_global_dnn_feats_fc7';
inputData.inputDir = baseDir;
load ~/code/mircs/s40_fra
test_images = {s40_fra.imageID};
inputData.fileList = struct('name',test_images);
testing = false;
checkIfNeeded = false;
run_all_2(inputData,outDir,'global_dnn_feats',testing ,[],'mcluster03',checkIfNeeded,[]);


%% move temporarily the fra_db images to another directory
landmarkDir = '~/storage/s40_my_facial_landmarks';
tmp_landmarkDir = '~/storage/s40_my_facial_landmarks_tmp';

load fra_db;
ensuredir(tmp_landmarkDir);
for u = 1:length(fra_db)
    from = j2m(landmarkDir,fra_db(u));
    to = j2m(tmp_landmarkDir,fra_db(u));
    movefile(from,to);
end
