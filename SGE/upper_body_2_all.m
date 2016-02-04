
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_upper_body_2';
inputData.inputDir = baseDir;
suffix =[];testing = false;
% load ~/code/mircs/faceActionImageNames;

% for t = 1:length(faceActionImageNames)
%     faceActionImageNames{t} = fullfile(baseDir,faceActionImageNames{t});
% end
% inputData.fileList = struct('name',faceActionImageNames);
% inputData = rmfield(inputData,'fileList');
run_all_2(inputData,outDir,'upper_body_2_parallel',testing ,suffix,'mcluster03',false,[])

%% willow actions.
baseDir = '~/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages/';
outDir = '/home/amirro/storage/uiuc_upper_body';
inputData.inputDir = baseDir;
justTesting = false;
run_all_2(inputData,outDir,'upper_body_2_parallel',justTesting);
