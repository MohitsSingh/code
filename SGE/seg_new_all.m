
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_seg_new';
inputData.inputDir = baseDir;
suffix =[];testing = true;
% load ~/code/mircs/faceActionImageNames;

% for t = 1:length(faceActionImageNames)
%     faceActionImageNames{t} = fullfile(baseDir,faceActionImageNames{t});
% end
% inputData.fileList = struct('name',faceActionImageNames);

run_all_2(inputData,outDir,'seg_new_parallel',testing ,suffix);

%% run on the fra_db regions as well. 

%%

load ~/code/mircs/fra_db.mat
imgsDir = '~/storage/data/Stanford40/JPEGImages/'
%params = struct('name',
params = struct('name',cellfun2(@(x) fullfile(imgsDir,x),{fra_db.imageID}));
outDir = '~/storage/s40_seg_new';
inputData.inputDir = baseDir;
suffix =[];testing = false;
checkIfNeeded = false;
run_all_3(params,outDir,'seg_new_parallel',testing,'mcluster03');

