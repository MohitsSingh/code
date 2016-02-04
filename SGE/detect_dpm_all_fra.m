
%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_fra_dpm';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/fra_db.mat;

for t = 1:length(fra_db)
    imgs{t} = fullfile(baseDir,fra_db(t).imageID);
end
inputData.fileList = struct('name',imgs);
run_all_2(inputData,outDir,'detect_dpm_parallel_fra',testing ,suffix);