
%%
baseDir = '/home/amirro/code/ruining_context/data/voc07/VOCdevkit/VOC2007/JPEGImages/';
outDir = '/home/amirro/code/ruining_context/amir_exp';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load /home/amirro/code/ruining_context/imdb.mat;
test_images = imdb.images.name([imdb.images.set]==3);
for t = 1:length(test_images)
    test_images{t} = fullfile(baseDir,test_images{t});
end
inputData.fileList = struct('name',test_images);
run_all_2(inputData,outDir,'pascal_features_parallel',testing ,suffix);