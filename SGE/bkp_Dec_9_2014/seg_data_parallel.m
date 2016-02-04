function seg_data_parallel(baseDir,d,indRange,outDir)
cd ~/code/mircs;
initpath;
config;
load ~/storage/misc/imageData_new;
addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));
addpath(genpath('~/code/3rdparty/geom2d'));
[learnParams,conf] = getDefaultLearningParams(conf,1024);
fisherFeatureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,currentID);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    fprintf('calculating for: %s!\n',filename);
    imageIndex = findImageIndex(newImageData,currentID);
    curImageData = newImageData(imageIndex);
    if (curImageData.faceScore < -.6)
        fprintf('face score too low! skipping ... \n');
        continue;
    end
    segData = getSegData_debug(conf,curImageData);
    isValid = true;
    if (max(segData.totalScores(:)) < -10)
        disp('skipping since no good candidates were found');
        isValid=false;
        fisherFeats = []; curSubIm = []; fisherFeats_reg = [];
    else
        
        %     sub_im = extractObjectWindow(conf,curImageData,segData,false);
        [curSubIm,windowPoly,orig_poly,target_rect] = extractObjectWindow(conf,curImageData,segData,false);
        I = getImage(conf,curImageData);
        [gc_segResult,obj_box] = checkSegmentation(I,windowPoly,orig_poly);
        regs = shiftRegions(gc_segResult,round(obj_box),I);
        reg_sub = rectifyWindow(regs,windowPoly,target_rect);
        if (~isempty(curSubIm))
            fisherFeats = fisherFeatureExtractor.extractFeatures(curSubIm); %#ok<NASGU>
            fisherFeats_reg = fisherFeatureExtractor.extractFeatures(curSubIm,reg_sub); %#ok<NASGU>
        else
            fisherFeats = [];%#ok<NASGU>
            isValid = false;%#ok<NASGU>
        end
    end
    save(resFileName,'segData','curSubIm','fisherFeats','fisherFeats_reg','isValid');
    fprintf('\tdone witn current image!\n');
end
fprintf('\n\nFinished all images in current batch\n\n\n!\n');