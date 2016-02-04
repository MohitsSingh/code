
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_sal_fine';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/faceActionImageNames;

% for t = 1:length(faceActionImageNames)
%     faceActionImageNames{t} = fullfile(baseDir,faceActionImageNames{t});
% end
% inputData.fileList = struct('name',faceActionImageNames);

run_all_2(inputData,outDir,'foreground_saliency_parallel',testing,suffix);