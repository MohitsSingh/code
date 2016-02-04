if (0)
%%%%%% Experiment 33 %%%%%%%
% March. 30, 2014

% make a detector for different facial parts, based on pose, which will
% enable you to score them based on appearance and hence tell if they are
% present or not.

% function experiment_0033
% initpath;
% config;
resultDir = ['~/mircs/experiments/' mfilename];
% ensuredir(resultDir);
% addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
% addpath('/home/amirro/code/3rdparty/sliding_segments');
L_imgs = load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');
L_pts = load('/home/amirro/mircs/experiments/experiment_0008/ptsData');
L_pts.ptsData = L_pts.ptsData(1:2:end);
L_pts.poses = L_pts.poses(1:2:end);
L_pts.ellipses= L_pts.ellipses(1:2:end);
rolls = [L_pts.poses.roll];
pitch = [L_pts.poses.pitch];
yaw = [L_pts.poses.yaw];
poses = L_pts.poses;
poseVecs = [rolls;pitch;yaw];
% poseVecs = [poses.yaw];
% [n,x] = hist(poseVecs,10);
% cluster according to poses....
[IDX_pose,C_pose] = kmeans2(poseVecs',10,struct('nTrial',10,'display',1));
save ~/mircs/experiments/common/poseVecs.mat poseVecs IDX_pose C_pose

baseDir = '~/storage/train_faces/aflw_cropped_context/';
d = dir(fullfile(baseDir,'*.jpg'));
pos_train_images = d(1:2:end);

%%
% u = unique(IDX_pose);
all_lips = {};
n = 0;
% for iu = 1:length(u)
%     sel_ = find(IDX_pose == u(iu));
%     for ik = 1:length(sel_)
%         ik

for k = 1:length(pos_train_images)
    n = n+1
    %k = sel_(ik);
    %         if (IDX_pose(k)==iu)
    I = imread(fullfile(baseDir, pos_train_images(k).name));
    %         I = I(round(.25*end:.75*end),round(.25*end:.75*end),:);
    w0 = size(I,1)*.25;
    orig_coords = ((2*w0/80))*L_pts.ptsData(k).pts+w0;
    
    ptNames = L_pts.ptsData(k).pointNames;
    mLeft = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthLeftCorner'));
    mRight = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthRightCorner'));
    mCenter = cellfun(@(x) ~isempty(x),strfind(ptNames,'MouthCenter'));
    
    if (any(mCenter)) % prefer center over left or right
        p = orig_coords(mCenter,:);
    elseif (any(mLeft))
        p = orig_coords(mLeft,:);
    elseif (any(mRight))
        p = orig_coords(mRight,:);
    end
    p = inflatebbox([p p],[40 40]*size(I,1)/160,'both',true);
    tt = cropper(I,round(p));
    tt = imResample(tt,[40 40]);
    all_lips{end+1} = tt;
    % %             %clf; imagesc(L_imgs.ims{k}); axis image; hold on;
%     clf; imagesc(tt); axis image; hold on;drawnow;pause;
    %         showCoords();
    %         end
end

%% 
% yoyo = yaw(1:1000);
% [y,iy] = sort(yoyo);
% displayImageSeries_simple(all_lips(iy(1:10:end)),.01);
% 

end

% save ~/storage/all_lips.mat all_lips 

%%

sel_ = abs(rolls*180/pi) < 10 & abs(pitch*180/pi) < 10;
poseVecs = poseVecs(:,sel_);
poseVecs = poseVecs(3,:);
% poseVecs = [poses.yaw];
% [n,x] = hist(poseVecs,10);
% cluster according to poses....
[IDX_pose,C_pose] = kmeans2(poseVecs',10,struct('nTrial',10,'display',1));
X = fevalArrays(cat(4,all_lips{sel_}),@(x)col(fhog(im2single(x),4)));
[IDX,C] = kmeans2(X',7,struct('nTrial',1,'display',1,'outFrac',.1));
maxPerCluster = .5;
% make the clustering consistent between different runs...
% clustersPath = '~/mircs/experiments/common/lipClusters_big.mat';
IDX = IDX_pose; C = C_pose;
% if (~exist(clustersPath,'file'))
[clusters,ims_,inds] = makeClusterImages(all_lips,C',IDX',[],[],.9);
displayImageSeries_simple(ims_);
[d,id] = sort(IDX);
plot(180*poseVecs(id)/pi);
aa = all_lips(sel_);

% showSorted(all_lips,yaw,50);

mImage(aa(IDX==1));

% for each lip cluster, make a detector.
%%
for iCluster = 1:length(inds)
    posImagesDir = sprintf('~/storage/faceTraining/posLips_full_%03.0f',iCluster);
    ensuredir(posImagesDir);
    negImagesDir = '~/storage/faceTraining/noFaces_small';
    multiWrite(all_lips(inds{iCluster}),posImagesDir);
    
    opts = acfTrain();
    opts.name = sprintf('Lips_full%03.0f',iCluster);
    opts.posWinDir = posImagesDir;
    opts.negImgDir = negImagesDir;
    opts.modelDs = [40 40];
    opts.modelDsPad = [40 40];
    opts.pPyramid.pChns.shrink = 5;
    opts.pPyramid.pChns.pColor.enabled = 1;
    opts.pPyramid.pChns.pColor.colorSpace = 'rgb';
    opts.pPyramid.pChns.pGradHist.enabled = 1;
    opts.pPyramid.pChns.pGradMag.enabled = 1;
    opts.pPyramid.pChns.pCustom(1).enabled = 1;
    opts.pPyramid.pChns.pCustom.name = 'fhog';
    opts.pPyramid.pChns.pCustom.hFunc = @fhog;
    opts.pPyramid.pChns.pCustom.pFunc = {5};
    opts.pPyramid.pChns.pCustom.padWith = 0;
    opts.nNeg = 2000;
    opts.nAccNeg = 10000;
    opts.nWeak = [128 256 1024];
    detectors{iCluster} = acfTrain(opts);
end
detectors{1}.opts.pNms.separate = 1;
save ~/mircs/experiments/experiment_0033/detectors_full.mat detectors

%%
imshow(I)
bbs = acfDetect(I,detectors)
%%

for k = 1:length(pos_train_images)
    k
%     if (IDX_pose(k)~=1),continue;end
    I = imread(fullfile(baseDir, pos_train_images(k).name));
    bbs = {};
%     for t = 1:length(detectors)
%         bbs{t} = 
%     end
    bbs = acfDetect(I,detectors);
    
%     bbs = cat(1,bbs{:});
    if (~isempty(bbs))
        
        [r,ir] = sort(bbs(:,5),'descend');bbs = bbs(ir(1),:);
        bbs(ir(1),:)
        bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);    
        clf; imagesc(I); axis image; hold on; plotBoxes(bbs,'g','LineWidth',2);drawnow; pause
    end
end

%% apply the detector to each mouth region in the imagedata
load ~/storage/misc/imageData_new;
load ~/mircs/experiments/experiment_0033/detectors.mat 
detectors{1}.opts.pNms.separate=1;

%%
%%
lipScores = -inf(size(newImageData));
%%
end

%%

for k = 1:length(newImageData)
    k
    if (mod(k,100)==0) ,disp(k);end
    curImageData = newImageData(k);
    %
    if (newImageData(k).faceScore >=-.6)
        if ~(~newImageData(k).label && k > 4000), continue; end
        [M,landmarks,face_box,face_poly] = getSubImage(conf,curImageData,1,true);        
        ddd = 100;
        M = imResample(M,ddd*[1 1],'bilinear');
        curScore = -inf;
        for rot = -30:10:30
            M_rot = imrotate(M,rot,'bilinear','crop');
            bbs = acfDetect(M_rot,detectors);
            %     bbs = cat(1,bbs{:});
            if (~isempty(bbs))
                curScore = max(curScore,max(bbs(:,5)));
                [b,ib] = sort(bbs(:,5),1,'descend');bbs = bbs(ib(1),:);
%                 bbs
                bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
%                                 clf; imagesc(M_rot); axis image; hold on; plotBoxes(bbs,'g','LineWidth',2);drawnow; pause
            end
        end
        lipScores(k) = curScore;
        clf; imagesc(M); axis image; hold on; title(num2str(curScore)); drawnow; pause;
                curScore
    end    
end


%% after having run everything, load the lip scores for each image.



save lipScores_new lipScores



% acfDetect
% s
