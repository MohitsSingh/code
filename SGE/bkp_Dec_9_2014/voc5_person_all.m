
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_voc5_person';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/faceActionImageNames;

for t = 1:length(faceActionImageNames)
    faceActionImageNames{t} = fullfile(baseDir,faceActionImageNames{t});
end
inputData.fileList = struct('name',faceActionImageNames);

run_all_2(inputData,outDir,'voc5_person_parallel',testing ,suffix);
