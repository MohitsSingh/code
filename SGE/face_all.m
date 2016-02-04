close all; clear all
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
% outDir = '/home/amirro/storage/faces_s40_big_x2_new';
outDir = '/home/amirro/storage/faces_s40_big_x2_new_rot';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'face_parallel',false);


%% phrasal recognition...
close all; clear all
baseDir = '~/storage/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages';
outDir = '/home/amirro/storage/faces_phrasal';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'face_parallel',false);

%% 
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_upper_body_faces';
inputData.inputDir = baseDir;
load ~/storage/misc/upper_bodies.mat;
inputNames = {data.imageID};
inputData.fileList = struct('name',inputNames);

suffix =[];testing = false;
run_all_2(inputData,outDir,'face_parallel_new',testing,suffix);

%% extract features from pascal...
VOCinit;

% train and test classifier for each action

for i=2:VOCopts.nactions % skip "other"
    cls=VOCopts.actions{i};    
    [imgids,objids,classifier.gt]=  textread(sprintf(VOCopts.action.clsimgsetpath,cls,VOCopts.trainset),'%s %d %d');
