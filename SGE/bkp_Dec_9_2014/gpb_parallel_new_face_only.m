function gpb_parallel_new_face_only(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;config;
handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';
addpath(handDetectionBaseDir );
addpath(fullfile(handDetectionBaseDir,'code/globalPb'));
addpath('/home/amirro/code/utils');
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath('/home/amirro/code/mircs/utils/');
load ~/storage/misc/imageData_new;
% addpath('~/code/mircs');
for k = 1:length(indRange)
    currentID=d(indRange(k)).name;
    %fprintf(['current image: ' currentID '\n']);
    resPath = j2m(outDir,currentID);
    imageIndex = findImageIndex(newImageData,currentID);
    curImageData = newImageData(imageIndex);
    res = [];
    if (curImageData.faceScore < -.6)
        save(resPath,'res');
        %fprintf(2,['skipping ' currentID ' due to low face score\n']);
        %drawnow('update');
        continue;
    end
    %     drawnow('update');
    
    [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = getSubImage(conf,curImageData,1.5,true);
    M = imResample(M,200/size(M,1),'bilinear');
    %fclose(fopen([resPath '.started'],'a'));
    loadOrCalc(conf,@gpb_segmentation,M,resPath);
    %fclose(fopen([resPath '.finished'],'a'));
    %fprintf(2,'done with : %s\n:', currentID);
end
fprintf('\n\n ***** finished all files in batch ****\n\n\n\n');
end