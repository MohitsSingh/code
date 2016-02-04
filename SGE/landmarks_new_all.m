clear all
inputData.inputDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/landmarks_s40_new_big';
run_all(inputData,outDir,'landmarks_new_parallel',false);

%% phrasal recognition....
close all; clear all
baseDir = '/home/amirro/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages';
outDir = '/home/amirro/storage/landmarks_phrasal';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'landmarks_new_parallel',false);

%% run on aflw faces. 
close all; clear all;

L_imgs = load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');
inputData.inputDir = 'fd';
inputData.images = L_imgs.ims;

outDir = '/home/amirro/storage/landmarks_aflw';
run_all(inputData,outDir,'landmarks_new_parallel_ims',false);



%%
% run with multipie-independent
inputData.inputDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/landmarks_s40_new_big_2';
run_all(inputData,outDir,'landmarks_new_parallel',false);
