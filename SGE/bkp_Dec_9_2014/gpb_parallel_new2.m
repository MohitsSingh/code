function gpb_parallel_new2(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;config;
handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));
addpath('/home/amirro/code/utils');
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath('/home/amirro/code/mircs/utils/');
% addpath('~/code/mircs');
for k = 1:length(indRange)    
    resPath = j2m(outDir,d(indRange(k)).name);
    [I,I_rect] = getImage(conf,d(indRange(k)).name);
    loadOrCalc(conf,@gpb_segmentation,I,resPath);
    fprintf('done with : %s\n:', d(indRange(k)).name);
end
fprintf('\n\n ***** finished all files in batch ****\n\n\n\n');
end