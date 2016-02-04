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
% end
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
suffix =[];testing = true;
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

%% run new feature extraction pipeline
outDir = '~/storage/s40/feature_pipeline';
testing = false;
clear inputData;
testing = true;
test_images = {s40_fra.imageID};
inputData = struct('name',test_images);

testing = false;
run_all_3(inputData,outDir,'action_feature_extraction',testing,'mcluster03');

%%

% 1. detect all faces on the aflw_cropped dataset
% 2. for each face detect landmarks with zhu
% 3. store all landmarks...

% new face detection, for pose estimation.
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;
test_images = {fra_db.imageID};
inputData.fileList = struct('name',test_images);
run_all_3(inputData,outDir,'fra_faces_baw',testing,suffix,'mcluster03');


%% extract convnet features from aflw images

%~/storage/misc/kp_pred_data.mat

params = struct('name',{});
facesDir = '~/storage/aflw_face_imgs'; ensuredir(facesDir);
for u = 1:length(curImgs)
    if (mod(u,100)==0)
        disp(u/length(curImgs))
    end
    %params(u).img = curImgs{u};
    params(u).name = num2str(u);
    img = curImgs{u};
    %     save(fullfile(facesDir,[num2str(u) '.mat']),'img');
end

outDir = '~/storage/aflw_face_deep_features';

suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs',testing,'mcluster03');

%% do the same for the sub images of fra_db
params = struct('name',{});
facesDir = '~/storage/fra_db_face_imgs'; ensuredir(facesDir);
for u = 1:length(subImgs)
    if (mod(u,100)==0)
        disp(u/length(subImgs))
    end
    
    img = subImgs{u};
    if (isempty(img))
        continue
    end
    save(fullfile(facesDir,[num2str(u) '.mat']),'img');
    params(u).name = num2str(u);
    params(u).faceDir = facesDir;
end

goods = false(size(params));
for u = 1:length(params)
    if (~isempty(params(u).name))
        goods(u) = true;
    end
end
params = params(goods);
outDir = '~/storage/fra_db_face_deep_features';

suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs',testing,'mcluster03');


%%
%% find faces on microsoft coco
params = struct('name',{});
%inputDir = '~/storage/mscoco/train2014';
%outDir = '~/storage/mscoco/train_faces';ensuredir(outDir);
inputDir = '~/storage/mscoco/val2014';
outDir = '~/storage/mscoco/val_faces';ensuredir(outDir);
f = dir(fullfile(inputDir,'*.jpg'));
for u = 1:length(f)
    if (mod(u,100)==0)
        disp(u/length(f))
    end
    params(u).name = fullfile(f(u).name);
    params(u).path = fullfile(inputDir,f(u).name);
    params(u).getDeepFeats = true;
end
addpath('~/code/utils');
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');

imgs_and_faces_val = collectFaceDetections(params,inputDir,outDir);
%%
params = struct('name',{});
%inputDir = '~/storage/mscoco/train2014';
%outDir = '~/storage/mscoco/train_faces';ensuredir(outDir);
inputDir = '~/storage/mscoco/train2014';
outDir = '~/storage/mscoco/train_faces';ensuredir(outDir);
f = dir(fullfile(inputDir,'*.jpg'));
for u = 1:length(f)
    if (mod(u,100)==0)
        disp(u/length(f))
    end
    params(u).name = fullfile(f(u).name);
    params(u).path = fullfile(inputDir,f(u).name);
    params(u).getDeepFeats = true;
end
addpath('~/code/utils');
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
imgs_and_faces_train = collectFaceDetections(params,inputDir,outDir);
img_inds = {};
boxes = {};
for t = 1:length(imgs_and_faces_train)
    %     t
    boxes{t} = imgs_and_faces_train(t).faces;
    img_inds{t} = ones(size(boxes{t},1),1)*t;
end
%%

addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
imgs_and_faces = collectFaceDetections(params,inputDir,outDir);

% save imgs_and_faces.mat imgs_and_faces

img_inds = {};
boxes = {};
for t = 1:length(imgs_and_faces)
    boxes{t} = imgs_and_faces(t).faces;
    img_inds{t} = ones(size(boxes{t},1),1)*t;
end
all_img_paths = {imgs_and_faces.path};
load ~/non_person_paths.mat
is_non_person = ismember(all_img_paths,non_person_paths);
boxes = cat(1,boxes{:});
img_inds = cat(1,img_inds{:});
[u,iu] = sort(boxes(:,end),'descend');

%%
face_det_imgs = {};
close all
figure(1)
jump_ = 3;
has_people = {};
tic_id = ticStatus('cropping face detections',.5,.5);
for k = 1:jump_:length(u)
    %     k
    %k = iu(ik);
    %     if (~is_non_person(img_inds(k)))
    %         continue
    %     end
    
    %         k = ik
    bbox = boxes(k,:);
    %     if (bbox(end)<-.2)
    %         continue
    %     end
    % get only negative examples...
    if (~is_non_person(img_inds(k)))
        continue
    end
    has_people{end+1} = false;
    
    imgPath = imgs_and_faces(img_inds(k)).path;
    I = imread(imgPath);
    %     clf;imagesc(I); drawnow;pause
    face_det_imgs{end+1} = cropper(I,round(bbox));
    tocStatus(tic_id,k/length(u));
    %     continue
end

save ~/storage/misc/false_face_det_images.mat face_det_imgs
% vals = u(1:jump_:end);
%%

%%
%%
%clf; imagesc2(imread(imgPath));
plotBoxes(bbox);
title(num2str(bbox(end)));
drawnow;
%     pause
pause(.1)
% end


%%
%% find faces in non-faces images.
%cd ~/code/mircs/
load ~/storage/misc/non_person_paths
params = struct('path',non_person_paths);
inputDir = [];
for u = 1:length(params)
    if (mod(u,100)==0)
        disp(u/length(params))
    end
    [~,name,ext] = fileparts(non_person_paths{u});
    params(u).name = [name ext];
end

addpath('~/code/utils');
outDir = '~/storage/voc_non_faces';ensuredir(outDir);

suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
imgDir = '/home/amirro/storage/data/VOCdevkit/VOC2012/JPEGImages/';
faces_and_images = collectFaceDetections([],imgDir,outDir);

%%

imgs_and_faces = struct('path',{},'faces',{});
m = length(params)
for t = 1:m
    t
    imgs_and_faces(t).path = params(t).path;
    if (exist(j2m(outDir,params(t).name),'file'))
        L = load(j2m(outDir,params(t).name));
        imgs_and_faces(t).faces = L.detections.boxes;
        
        %         if (~isempty(L.detections.boxes))
        %             imgPath = imgs_and_faces(t).path;
        %             clf; imagesc2(imread(imgPath));
        %             plotBoxes(L.detections.boxes);
        %             pause
        %         end
    end
end

img_inds = {};
boxes = {};
for t = 1:m
    boxes{t} = imgs_and_faces(t).faces;
    img_inds{t} = ones(size(boxes{t},1),1)*t;
end

boxes = cat(1,boxes{:});
img_inds = cat(1,img_inds{:});

[u,iu] = sort(boxes(:,end),'descend');
%%
close all
figure(1)
jump_ = 100;
for ik = 1:jump_:length(imgs_and_faces_train)
    ik
    k = iu(ik);
    %     k = ik
    bbox = boxes(k,:)
    %     if (bbox(end)>0
    %         continue
    %     end
    imgPath = imgs_and_faces_train(img_inds(k)).path;
    
    clf; imagesc2(imread(imgPath));
    plotBoxes(bbox);
    title(num2str(bbox(end)));
    drawnow;
    %     pause
    pause(.1)
end


%% run zhu on aflw images...
%cd ~/code/mircs/
%load ~/storage/misc/non_person_paths
load ~/storage/misc/aflw_with_pts.mat ims pts poses scores inflateFactor resizeFactors;
params = struct('name',cellfun2(@num2str, mat2cell2(1:length(ims),[1,length(ims)])),'img',ims);
inputDir = [];
addpath('~/code/utils');
outDir = '~/storage/aflw_zhu';ensuredir(outDir);
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'faces_zhu',testing,'mcluster03');
%%
load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
s40_fra = s40_fra_faces_d
params = struct('name',{s40_fra.imageID});
outDir = '~/storage/s40_dnn_feats_deep';
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs_2',testing,'mcluster03');
all_dnn_feats_deep = load_all('extract_dnn_feats_imgs_2',params,outDir);
all_dnn_feats_deep = cat(2,all_dnn_feats_deep{:});
% addpath('~/code/3rdparty/serialization');
%B = hlp_serialize(all_dnn_feats);
save ~/storage/misc/all_dnn_feats_deep.mat all_dnn_feats_deep -v7.3

%% extract dnn feats from full and cropped images
load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
s40_fra = s40_fra_faces_d
params = struct('name',{s40_fra.imageID});
outDir = '~/storage/s40_dnn_feats';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs_2',testing,'mcluster03');

all_dnn_feats = load_all('extract_dnn_feats_imgs_2',params,outDir);
all_dnn_feats = cat(2,all_dnn_feats{:});
cd
addpath('~/code/3rdparty/serialization');
%B = hlp_serialize(all_dnn_feats);
save ~/storage/misc/all_dnn_feats.mat all_dnn_feats -v7.3
%

%% extract dnn feats from faces (annotated for ground truth)
load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
s40_fra = s40_fra_faces_d
params = struct('name',{s40_fra.imageID});
outDir = '~/storage/s40_dnn_feats_faces';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs_2_faces',testing,'mcluster03');
all_dnn_feats_faces = load_all('extract_dnn_feats_imgs_2_faces',params,outDir);
all_dnn_feats_faces = cat(2,all_dnn_feats_faces{:});
save ~/storage/misc/all_dnn_feats_faces.mat all_dnn_feats_faces -v6
%
%% get edge boxes from non-person images....
load ~/code/mircs/non_person_paths_val.mat;
names = {};
for t = 1:length(non_person_paths)
    names{t} = non_person_paths{t}(26:end);
end
params = struct('name',names,'path',non_person_paths);
outDir = '~/storage/coco_non_person_boxes';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_edge_boxes',testing,'mcluster03');

all_boxes = load_all('extract_edge_boxes',params,outDir);

save ~/storage/misc/coco_val_edge_boxes.mat all_boxes -v7.3
%%
%% run the face detector on all the stanford 40 images
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
names = {fra_db.imageID};
paths = cellfun2(@(x)  fullfile(baseDir,x),names);
params = struct('name',names,'path',paths);
outDir = '~/storage/s40_faces_baw';
clear inputData;
inputData.inputDir = baseDir;
suffix =[];testing = true;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
testing = false;
run_all_3(params,outDir,'faces_baw',testing,'mcluster01');
all_faces = load_all('faces_baw',params,outDir);
all_faces = [all_faces{:}];
save ~/storage/misc/s40_face_dets.mat all_faces
%%
for t = 801:length(fra_db)
    I = getImage(conf,fra_db(t));
    curFaces = all_faces(t);
    cur_boxes = {};
    for u = 1:length(curFaces.detections);
        curFaces.detections(u).boxes(:,end+1) = curFaces.detections(u).rot;
        cur_boxes{end+1} = curFaces.detections(u).boxes;
    end
    cur_boxes = cat(1,cur_boxes{:});
    [z,iz] = sort(cur_boxes(:,6),'descend');
    for k = 1:3
        U = imrotate(I,cur_boxes(iz(k),7),'bilinear','crop');
        U(U<0) = 0;U(U>1) = 1;
        
        clf; imagesc2(U);
        plotBoxes(cur_boxes(iz(k),:));
        pause
    end
%     pause
end    
%% my nn-based facial landmarks.
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
names = {fra_db.imageID};
% names = {'cleaning_the_floor_010.jpg'};
paths = cellfun2(@(x)  fullfile(baseDir,x),names);
f = dir(outDir);
f = f(3:end);
f = {f.name};
f = cellfun2(@(x) [x(1:end-4) '.jpg'],f);
newNames = setdiff(names,f);
% names = {fra_db.imageID};
% names = {'cleaning_the_floor_010.jpg'};
newPaths = cellfun2(@(x)  fullfile(baseDir,x),newNames);
params = struct('name',names,'path',paths);
% params = struct('name',newNames,'path',newPaths);
outDir = '~/storage/s40_facial_lm_global';
clear inputData;
inputData.inputDir = baseDir;
suffix =[];
testing = false;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_3(params,outDir,'facial_lm_global',testing,'mcluster03');
face_lm_global = load_all('facial_lm_global',params,outDir);
face_lm_global= [face_lm_global{:}];
save ~/storage/misc/s40_face_lm_global.mat face_lm_global

%% extract face related features

%%
load ~/storage/misc/s40_fra_faces_d_new;
fra_db = s40_fra_faces_d;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
names = {fra_db.imageID};
% names = {'cleaning_the_floor_010.jpg'};
paths = cellfun2(@(x)  fullfile(baseDir,x),names);
% f = dir(outDir);
outDir = '~/storage/s40_head_feats_deep_05';
load ~/storage/misc/s40_face_lm_global.mat; % face_lm_global
load ~/storage/misc/s40_face_detections.mat; % all_detections
detections =  [all_detections.detections];
boxes = {detections.boxes};
kp_global = {face_lm_global.kp_global};
params = struct('name',names,'path',paths,'boxes',boxes,'kp_global',kp_global,'partOfHead',.5);
testing = false;
run_all_3(params,outDir,'extract_head_feats',testing,'mcluster03');

all_dnn_feats_head = load_all('extract_head_feats',params,outDir);
all_dnn_feats_head = cat(2,all_dnn_feats_head{:});

save ~/storage/misc/all_dnn_feats_head_05.mat all_dnn_feats_head -v6
% x2(getImage(conf,fra_db(5)));
% plotBoxes(boxes{5}(1,:))
%% pascal actions!!



% imdb
params = struct('name',{});
outdir = '~/storage/pas_feats';mkdir(outdir);
for u = 1:length(imdb)
    params(u).name = [imdb(u).image_id '_' num2str(imdb(u).idx)];
    params(u).imgData = imdb(u);
end

testing = false;
run_all_3(params,outdir,'feat_pipeline',testing,'mcluster03');

all_dnn_feats_head = load_all('extract_head_feats',params,outDir);
all_dnn_feats_head = cat(2,all_dnn_feats_head{:});

%%
names = {s40_fra_faces_d.imageID};
paths = cellfun2(@(x)  fullfile(baseDir,x),names);
params = struct('name',names,'path',paths);
% params = struct('name',newNames,'path',newPaths);
outDir = '~/storage/fra_faces_baw_new';
testing = false;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
s40_face_dets = load_all('faces_baw',params,outDir);
s40_face_dets = cat(2,s40_face_dets{:});
save ~/storage/misc/s40_face_dets.mat s40_face_dets
% summarize the face detection results...
%%
addpath('~/code/mircs/');
load ~/code/mircs/imgSizes.mat
s40_person_face_dets = consolidateFaceDetections(s40_fra_faces_d,s40_face_dets);



% % save ~/storage/misc/s40_face_dets2.mat s40_person_face_dets
%% do the same for the drinking-extended dataset...
imgsDir = '/home/amirro/data/drinking_extended/straw/';
d = dir(fullfile(imgsDir,'*.jpg'));
names = {d.name};
paths = cellfun2(@(x)  fullfile(imgsDir,x),names);
params = struct('name',names,'path',paths);
% params = struct('name',newNames,'path',newPaths);
outDir = '~/storage/faces_drinking_extended';
testing = true;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
skipMissing = true;
[extended_face_dets,goods] = load_all('faces_baw',params,outDir,skipMissing);
e_face_dets = consolidateFaceDetections(conf,paths(goods),cat(2,extended_face_dets{goods}));

%%

load ~/storage/data/drinking_dataset.mat
names = {};
paths = {drinking_dataset.imageID};
for t = 1:length(paths)    
    names{t} = sprintf('%05.0f',t);
end
params = struct('name',names,'path',paths);
% params = struct('name',newNames,'path',newPaths);
outDir = '~/storage/faces_drinking_extended_1';
testing = false;
% load ~/code/mircs/fra_db.mat;
% test_images = {fra_db.imageID};
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
skipMissing = true;
[extended_face_dets,goods] = load_all('faces_baw',params,outDir,skipMissing);
e_face_dets = consolidateFaceDetections(conf,paths(goods),cat(2,extended_face_dets{goods}));
extended_face_dets = cat(2,extended_face_dets{:});
save ~/storage/misc/extended_face_dets_1.mat e_face_dets



%%
params = struct('name',{});
% facesDir = '~/storage/aflw_face_imgs'; ensuredir(facesDir);
curImgs = all_ims_small;
for u = 1:length(curImgs)
    if (mod(u,100)==0)
        disp(u/length(curImgs))
    end
    %params(u).img = curImgs{u};
    params(u).name = num2str(u);
    img = curImgs{u};
    params(u).img = img;
    
    %     save(fullfile(facesDir,[num2str(u) '.mat']),'img');
end
outDir = '~/storage/aflw_face_deep_features2';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs',testing,'mcluster03');
all_faces_feats = load_all('extract_dnn_feats_imgs',params,outDir);
all_faces_feats = [all_faces_feats{:}]
aa = [all_faces_feats.feats_s];aa = cat(2,aa.x);

%%
load ~/storage/misc/neededIDS.mat
% ^params = struct('name',cellfun2(@(x) fullfile(conf.imgDir,x),{s40_fra([s40_fra.isTrain]).imageID}))
params = struct('name',cellfun2(@(x) fullfile(conf.imgDir,x),{s40_fra.imageID}))
% params = struct('name',cellfun2(@(x) fullfile(conf.imgDir,x),neededIDS))
outDir = '~/storage/s40_kriz_fc6_block_5_2';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_imgs_3',testing,'mcluster03');
all_faces_feats = load_all('extract_dnn_feats_imgs_3',params,outDir);
all_faces_feats = [all_faces_feats{:}]
aa = [all_faces_feats.feats_s];aa = cat(2,aa.x);


%%
load ~/storage/misc/fra_db_d.mat
params = struct('name',{fra_db_d.imageID}, 'img',{fra_db_d.I});
% params = struct('name',cellfun2(@(x) fullfile(conf.imgDir,x),neededIDS))
outDir = '~/storage/fra_db_face_seg_feats';
suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'extract_dnn_feats_masks',testing,'mcluster03');
all_faces_feats = load_all('extract_dnn_feats_masks',params,outDir);
all_faces_feats = [all_faces_feats{:}]
aa = [all_faces_feats.feats_s];aa = cat(2,aa.x);

%%
load ~/code/mircs/fra_db.mat
imgsDir = '~/storage/data/Stanford40/JPEGImages/'
%params = struct('name',
params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{fra_db.imageID}))
outDir = '~/storage/fra_db_seg_x2';
suffix =[];
testing = true;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'seg_new_parallel',testing,'mcluster03');
%all_faces_feats = load_all('seg_new_parallel',params,outDir);
%all_faces_feats = [all_faces_feats{:}]
%aa = [all_faces_feats.feats_s];aa = cat(2,aa.x);
% 
x2(imResample(I,.25))
%%
load ~/code/mircs/fra_db.mat
imgsDir = '~/storage/data/Stanford40/JPEGImages/'
%params = struct('name',
params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{fra_db.imageID}));
% params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{'blowing_bubbles_185.jpg'}));
for t = 1:length(params)
    params(t).aroundMouth = true;
end
outDir = '~/storage/fra_db_mouth_seg_x2';
d = dir(fullfile(outDir,'*.mat'));
setdiff({fra_db.imageID},cellfun2(@(x) strrep(x,'.mat','.jpg'), {d.name}))
suffix =[];
testing = true;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'seg_new_parallel',testing,'mcluster03');
%all_faces_feats = load_all('seg_new_parallel',params,outDir);
%all_faces_feats = [all_faces_feats{:}]
%aa = [all_faces_feats.feats_s];aa = cat(2,aa.x);
% 
% x2(imResample(I,.25))

%%

inputDir = '~/storage/data/faces/mtfl/AFLW';
a = getAllFiles(inputDir,'.jpg');
a = [a, getAllFiles('~/storage/data/faces/mtfl/net_7876','.jpg'),getAllFiles('~/storage/data/faces/mtfl/lfw5590','.jpg')];
names = {};
for t = 1:length(a)
    [~,name,ext] = fileparts(a{t});
    names{t} = [name ext];
end

%params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{fra_db(701).imageID}));
params = struct('path',a,'name',names);
for t = 1:length(params)
    params(t).aroundMouth = true;
end
outDir = '~/storage/mtfl_baw_faces';
suffix =[];
testing = false;
run_all_3(params,outDir,'faces_baw',testing,'mcluster03');
%%

kpParams.debug_ = false;
kpParams.wSize = wSize;
kpParams.extractHogsHelper = extractHogsHelper;
kpParams.im_subset = predData.im_subset;
kpParams.requiredKeypoints = requiredKeypoints;

for t = 7000:50:length(a)
    [~,name,ext] = fileparts(a{t});
    load(fullfile(outDir,[name '.mat']));
    I = imread(a{t});
    clf; imagesc2(I);
    faceBox = detections.boxes(1,:);
    plotBoxes(faceBox);   
    conf.get_full_image = true;
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;            
%     [rois,roiBox,I,scaleFactor] = get_rois_fra(conf,p,roiParams);
%     faceBox = rois(1).bbox;
    faceBox = inflatebbox(faceBox,1);
    bb = round(faceBox(1,1:4));
    kpParams.debug_ = true;
    [res.kp_global,res.kp_local] = myFindFacialKeyPoints(conf,I,bb,predData.XX,...
        predData.kdtree,predData.curImgs,predData.ress,predData.ptsData,kpParams);    
    dpc
end


%% full image segmentations
load ~/code/mircs/fra_db.mat
addpath(genpath('~/code/utils'));

FF = fra_db([fra_db.isTrain]);
imgsDir = '~/storage/data/Stanford40/JPEGImages/'
%params = struct('name',
params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{FF.imageID}));
% params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{'blowing_bubbles_185.jpg'}));
for t = 1:length(params)
    params(t).aroundMouth = true;
    params(t).full_image = true;
end
outDir = '~/storage/fra_db_seg_full';
d = dir(fullfile(outDir,'*.mat'));
remainingFiles = setdiff({FF.imageID},cellfun2(@(x) strrep(x,'.mat','.jpg'), {d.name}));

% params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{'phoning_246.jpg'}));
params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),remainingFiles));
% params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{'blowing_bubbles_185.jpg'}));
for t = 1:length(params)
    params(t).aroundMouth = true;
    params(t).full_image = true;
end

suffix =[];
testing = false;
%         inputs,outDir,fun,justTesting,cluster_name
run_all_3(params,outDir,'seg_new_parallel',testing,'mcluster03');


