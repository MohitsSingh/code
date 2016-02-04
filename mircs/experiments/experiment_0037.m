% Experiment 0037 - run shape sharing on all images
% 20/5/2014


% close all; clear all

resultDir = '~/storage/s40_shape_sharing';

cd ~/code/mircs
% mkdir(resultDir)
shapeSharingPath = '/net/mraid11/export/data/amirro/shapesharing';
load faceActionImageNames
imgDir = '~/data/Stanford40/JPEGImages';
cd(shapeSharingPath);
SetupPath;
while (true)
    try
        for k = 1:length(faceActionImageNames)
            filePaths{k} = fullfile(imgDir,faceActionImageNames{k});
        end
        
        T = randperm(length(filePaths));
        filePaths = filePaths(T);
        faceActionImageNames = faceActionImageNames(T);
        for k = 1:length(faceActionImageNames)
            k
            resPath = fullfile(resultDir,strrep(faceActionImageNames{k},'.jpg','.mat'));
            if (~exist(resPath,'file'))
                disp(resPath);
                [masks, timing] = ComputeSegment(filePaths{k});
                save(resPath,'masks','timing');
                disp('Done');
            end
        end
        
    catch me
    end
end