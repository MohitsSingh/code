%%%%%% Experiment 32 %%%%%%%
% March. 27, 2014

% make a new face detector using piotr dollar's acfTrain and many faces from AFLW.

function experiment_0032
initpath;
config;
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
baseDir = '~/storage/train_faces/aflw_cropped_context/';
d = dir(fullfile(baseDir,'*.jpg'));

qq = 2;
% % qq = 30;
pos_train_images = d(1:qq:end);
%
%%
ims = {};
for k = 1:length(pos_train_images)
    if (mod(k,100)==0)
        k
    end
    I = imread(fullfile(baseDir, pos_train_images(k).name));
    dd = .2;
    I = I(round(dd*end:(1-dd)*end),round(dd*end:(1-dd)*end),:);
    II = imResample(I,[80 80],'bilinear');
    if (mod(k,100)==0)
        %         k
        clf; imshow(II); drawnow;
    end
    %     size(I)
    if (length(size(II))==2)
        II = repmat(II,[1 1 3]);
    end
    ims{k} = II;
end
%%
load ~/mircs/experiments/common/poseVecs.mat
[clusters,ims_,inds] = makeClusterImages(ims,C_pose',IDX_pose',[],[],.9);
% displayImageSeries_simple(ims_);

posImagesDir = '~/storage/faceTraining/posFaces';
negImagesDir = '~/storage/faceTraining/noFaces';
for iCluster = 1:length(inds)
    posImagesDir = sprintf('~/storage/faceTraining/posFaces_big_full%03.0f',iCluster);
    ensuredir(posImagesDir);
    negImagesDir = '~/storage/faceTraining/noFaces';
    multiWrite(ims(inds{iCluster}),posImagesDir);
    opts = acfTrain();
    opts.name = sprintf('Faces_big__full%03.0f',iCluster);
    opts.posWinDir = posImagesDir;
    opts.negImgDir = negImagesDir;
    opts.modelDs = [72 72];
    opts.modelDsPad = [80 80];
    opts.pPyramid.pChns.shrink = 8;
    %     opts.modelDs = [48 48];
    %     opts.modelDsPad = [48 48];
    %     opts.pPyramid.pChns.shrink = 4;
    opts.pPyramid.pChns.pColor.enabled = 1;
    opts.pPyramid.pChns.pColor.colorSpace = 'rgb';
    opts.pPyramid.pChns.pGradHist.enabled = 1;
    opts.pPyramid.pChns.pGradMag.enabled = 1;
    opts.pPyramid.pChns.pCustom(1).enabled = 1;
    opts.pPyramid.pChns.pCustom.name = 'fhog';
    opts.pPyramid.pChns.pCustom.hFunc = @fhog;
    opts.pPyramid.pChns.pCustom.pFunc = {opts.pPyramid.pChns.shrink};
    opts.pPyramid.pChns.pCustom.padWith = 0;
    opts.nNeg = 2000;
    opts.nAccNeg = 10000;
    opts.nWeak = [128 256 1024 4096];
    detectors{iCluster} = acfTrain(opts);
end

detectors{1}.opts.pNms.separate=1;
save ~/mircs/experiments/experiment_0032/detectors_big_full.mat detectors

% negImagesDir = '~/storage/faceTraining/noFaces_small';
% multiWrite(ims,posImagesDir);

nonPersonIDS = getNonPersonIds(VOCopts);mkdir(negImagesDir);
% nonPersonIDS  = vl_colsubset(nonPersonIDS',500,'Uniform');
% for k = 1:length(nonPersonIDS)
%     k
%     outFile = fullfile(negImagesDir,sprintf('%010.0f.jpg',k));
%     I = getImage(conf,nonPersonIDS{k});
% %     clf;imagesc(I); axis image; drawnow
%     imwrite(I,outFile);
[test_ids,train_labels] = getImageSet(conf,'train',1,0);
load ~/storage/misc/imageData_new;

[all_bbs,all_ids] = piotr_detect(conf,detectors,{newImageData.imageID});

save piotr_res.mat all_bbs all_ids

%%
for k = 1:length(newImageData)
    k
    if (k<1 || ~newImageData(k).label),continue;end
    I = getImage(conf,newImageData(k).imageID);
    %bb = all_bbs{k};
    I_orig = imResample(I,2);
    for rots = 0;%-20:10:20
        I = imrotate(I_orig,rots,'bilinear','crop');
        bb = acfDetect(I,detectors);bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
        bb(bb(:,5)<40,:) = [];
        bb(:,end) = rots;
        %     s = bb(:,end);
        %     bb = bb(s>10,:);
        %     bb = bb(1:min(5,size(bb,1)),:);
        clf; imagesc(normalise(I)); axis image; hold on; plotBoxes(bb,'g','LineWidth',2);
        drawnow; pause;
    end
end
% bbs = cat(1,all_bbs);
%% vary the training parameters a bit.

baseDir = '~/storage/train_faces/aflw_cropped_context/';
d = dir(fullfile(baseDir,'*.jpg'));

qq = 2;
% % qq = 30;
pos_train_images = d(1:qq:end);
%
%%
ims = {};
for k = 1:length(pos_train_images)
    if (mod(k,100)==0)
        k
    end
    I = imread(fullfile(baseDir, pos_train_images(k).name));
    dd = .2;
    I = I(round(dd*end:(1-dd)*end),round(dd*end:(1-dd)*end),:);
    II = imResample(I,[56 56],'bilinear');
    if (mod(k,100)==0)
        %         k
        clf; imshow(II); drawnow;
    end
    %     size(I)
    if (length(size(II))==2)
        II = repmat(II,[1 1 3]);
    end
    ims{k} = II;
end
%%
load ~/mircs/experiments/common/poseVecs.mat
[clusters,ims_,inds] = makeClusterImages(ims,C_pose',IDX_pose',[],[],.9);
% displayImageSeries_simple(ims_);
%%
posImagesDir = '~/storage/faceTraining/posFaces';
negImagesDir = '~/storage/faceTraining/noFaces';
for iCluster = 1:length(inds)
    posImagesDir = sprintf('~/storage/faceTraining/posFaces_big_full_2%03.0f',iCluster);
    ensuredir(posImagesDir);
    negImagesDir = '~/storage/faceTraining/noFaces';
    multiWrite(ims(inds{iCluster}),posImagesDir);
    opts = acfTrain();
    opts.name = sprintf('Faces_big__full_2%03.0f',iCluster);
    opts.posWinDir = posImagesDir;
    opts.negImgDir = negImagesDir;
    opts.modelDs = [48 48];
    opts.modelDsPad = [56 56];
    opts.pPyramid.pChns.shrink = 4;
    opts.pPyramid.pChns.pColor.enabled = 0;
    opts.pPyramid.pChns.pColor.colorSpace = 'rgb';
    opts.pPyramid.pChns.pGradHist.enabled = 0;
    opts.pPyramid.pChns.pGradMag.enabled = 0;
    opts.pPyramid.pChns.pCustom(1).enabled = 1;
    opts.pPyramid.pChns.pCustom.name = 'fhog';
    opts.pPyramid.pChns.pCustom.hFunc = @fhog;
    opts.pPyramid.pChns.pCustom.pFunc = {opts.pPyramid.pChns.shrink};
    opts.pPyramid.pChns.pCustom.padWith = 0;
    opts.pBoost.pTree.fracFtrs = 1/16;
    opts.nNeg = 2000;
    opts.nAccNeg = 50000;
    opts.nWeak = [32 512 2048 8192];
    detectors{iCluster} = acfTrain(opts);
end

detectors{1}.opts.pNms.separate=1;
save ~/mircs/experiments/experiment_0032/detectors_big_full_2.mat detectors

%% one monolithic detector...
posImagesDir = '~/storage/faceTraining/posFaces_monolith';
ensuredir(posImagesDir);
negImagesDir = '~/storage/faceTraining/noFaces';
multiWrite(ims(1:10:end),posImagesDir);
opts = acfTrain();
opts.name = sprintf('Faces_monolith');
opts.posWinDir = posImagesDir;
opts.negImgDir = negImagesDir;
% checking if adding too many positives harms results.
opts.modelDs = [48 48];
opts.modelDsPad = [56 56];
opts.pPyramid.pChns.shrink = 4;
opts.pPyramid.pChns.pColor.enabled = 0;
opts.pPyramid.pChns.pGradHist.enabled = 0;
opts.pPyramid.pChns.pGradMag.enabled = 0;
opts.pPyramid.pChns.pCustom(1).enabled = 1;
opts.pPyramid.pChns.pCustom.name = 'fhog';
opts.pPyramid.pChns.pCustom.hFunc = @fhog;
opts.pPyramid.pChns.pCustom.pFunc = {opts.pPyramid.pChns.shrink};
opts.pPyramid.pChns.pCustom.padWith = 0;
opts.pBoost.pTree.fracFtrs = 1/32;

% check effects of number of positives, fracFtrs, modelDS vs modelDsPad,
% shrink... 
% 1000 images, many negs. shrink = 8; total failure. trying to reduce shrink size to 4.

opts.nNeg = 10000;
opts.nAccNeg = 100000;
opts.nWeak = [32 512 2048];
detector = acfTrain(opts);

%%


%% detect again on training images
for k = 1:length(newImageData)
    k
        %     if (k<1 || ~newImageData(k).label),continue;end
    I = getImage(conf,newImageData(k).imageID);
    % for k = 1:length(pos_train_images)
    % %     if (mod(k,100)==0)
    %         k
    % %     end
    %     I = imread(fullfile(baseDir, pos_train_images(k).name));
    %bb = all_bbs{k};
%         I = imResample(I,2);
    %     for rots = 0;%-20:10:20
%     %         I = imrotate(I_orig,rots,'bilinear','crop');
%     acfModify
    bb = acfDetect(I,detector);
    bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
    %         bb(bb(:,5)<40,:) = [];
%     bb(:,end) = rots;
%     s = bb(:,5);
%     [s,is] = max(s);
    %         if (s < 100)
    %             continue;
    %         end
    %         bb = bb(is,:);
    %     bb = bb(s>10,:);
    %     bb = bb(1:min(5,size(bb,1)),:);
    %         bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
    clf; imagesc(normalise(I)); axis image; hold on; plotBoxes(bb,'g','LineWidth',2);
    title(num2str(s));
    drawnow; pause;
end

