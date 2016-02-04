%%%%%% Experiment 12 %%%%%%%
% Nov. 17, 2013

% Tell if a facial keypoint is occluded or not.
% How to do it? train a classifier for this keypoint against random
% samples from other images; classifier score should be an indication.

initpath;
config;
L_pts = load('/home/amirro/mircs/experiments/experiment_0008/ptsData');
ptsData = L_pts.ptsData(1:2:end);
poses = L_pts.poses(1:2:end);
load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat'); % ims
yaw = 180*[poses.yaw]/pi;

poseVecs = [[poses.pitch];[poses.yaw];[poses.roll]];

% cluster according to poses....
[IDX_pose,C_pose] = kmeans2(poseVecs',10,struct('nTrial',1,'display',1));

% plot3(poseVecs(1,:),poseVecs(2,:),poseVecs(3,:),'ro')

% edges = linspace(-120,120,30);
% [n,bin] = histc(yaw,edges);
bar(linspace(-120,120,30),n);





%%
% load the zhu/ramanan results...
lipboxes = {};
tic;

for k = 1:length(ims)
    if (toc > 1)
        100*k/length(ims)
        tic
    end
    %     if (abs(yaw(k)) < 50)
    %         continue;
    %     end
%     L = load(sprintf('~/storage/landmarks_aflw/%05.0f.mat',k));
    L1 = landmarks(k);
    L.landmarks = L1;
    
   
    if (isempty(L.landmarks) || isempty(L.landmarks.s))
        continue;
    end
    
     if (0 && L.landmarks.s > 0 && size(L.landmarks.xy,1)==68)
         curIm = ims{k};
         clf;imagesc(curIm);
         hold on;
         bboxes = L.landmarks.xy;
         hold on; plotBoxes(bboxes);
         bc = boxCenters(bboxes);
         plot(bc(:,1),bc(:,2),'r-');
         for kk = 1:size(bc,1)
            text(bc(kk,1),bc(kk,2),num2str(kk),'Color','g');
         end
         
         inner_lips = [36 37 38 42 42 45 47 49 36];
         outer_lips = [35 34 33 32 39 40 41 44 46 51 48 50 35];
         plot(bc(inner_lips,1),bc(inner_lips,2),'y-','LineWidth',2);
         plot(bc(outer_lips,1),bc(outer_lips,2),'m-','LineWidth',2);
                           
        pause
     end
    
     if (L.landmarks.s > 0 && size(L.landmarks.xy,1)==39)
         curIm = ims{k};
         clf;imagesc(curIm);
         hold on;
         bboxes = L.landmarks.xy;
         hold on; plotBoxes(bboxes);
         bc = boxCenters(bboxes);
         plot(bc(:,1),bc(:,2),'r-');
         for kk = 1:size(bc,1)
            text(bc(kk,1),bc(kk,2),num2str(kk),'Color','g');
         end
         
         inner_lips = [25 24 26 23 27];
         outer_lips = [16:22];
         plot(bc(inner_lips,1),bc(inner_lips,2),'y-','LineWidth',2);
         plot(bc(outer_lips,1),bc(outer_lips,2),'m-','LineWidth',2);                           
        pause
    end
    
    landmarks(k) = L.landmarks;    
end
resDir = '~/mircs/experiments/experiment_0012';
mkdir(resDir);
save(fullfile(resDir,'landmarks.mat'),'landmarks');

empties = false(size(landmarks));
for k = 1:length(landmarks)
    if (isempty(landmarks(k).xy))
        empties(k) = true;
    end
end
ims = ims(~empties);
landmarks = landmarks(~empties);
IDX_pose = IDX_pose(~empties);

bboxes = zeros(size(landmarks,1),4);
for k = 1:length(landmarks)
    curLandmark = landmarks(k);
    if (size(curLandmark.xy,1) >= 51)
        lipCoords = boxCenters(curLandmark.xy(33:51,:));
        lipBox = pts2Box(lipCoords);
    elseif (size(curLandmark.xy,1) == 39)
        lipCoords = boxCenters(curLandmark.xy(16:22,:));
        lipBox = pts2Box(lipCoords);
    end
    bboxes(k,:) = lipBox;
end

bc = boxCenters(bboxes);
bboxes = inflatebbox(bboxes,[41 41],'both',true);
bboxes_c = round(clip_to_image(bboxes,[1 1 80 80]));
[rows cols] = BoxSize(bboxes_c);

goods = rows == 41 & cols == 41;
landmarks = landmarks(goods);
bin = bin(goods);
ims = ims(goods);
IDX_pose = IDX_pose(goods);
bboxes_c = bboxes_c(goods,:);

u = unique(bins);
scores = [landmarks.s];

T = -.5;
lipImages = multiCrop(conf,ims,bboxes_c);
save ~/mircs/experiments/experiment_0012/lipImages.mat lipImages

% make a detector according to clusters....
x = fevalArrays(cat(4,lipImages{:}),@(y) col(fhog(im2single(y),4)));
x(3101:end,:) = [];

% get some random patches from many images...
conf.features.vlfeat.cellsize = 4;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
% clusters = makeCluster(0,[]);

nPatches = 5000;
negSamples = sampleRandomPatches(conf,getNonPersonIds(conf.VOCopts),nPatches);

clear clusters;
for u = 1:size(C_pose,1)
    u
    
    
    curScores = scores(IDX_pose==u & scores' >= T);
    curImages = lipImages(IDX_pose==u & scores' >= T);
    
    % if more than 100 positive samples, throw out some
    [r,ir] = sort(curScores);
    curImages = curImages(ir);
    if (length(curScores) > 100)        
        curImages = curImages(1:100);        
    end
    
    conf.features.winsize = [10 10 31];
    x = imageSetFeatures2(conf,curImages,true,[]);
    clusters(u) = makeCluster(x,[]);
    
    
%     montage3(curImages);
%     pause
    
%     [ res , sizes] = imageSetFeatures2( conf,images,flat,sz)
    
%     pause;    
end

detectors = train_patch_classifier(conf,clusters,getNonPersonIds(conf.VOCopts),'suffix','lips');

% imagesc(hogDraw(reshape(detectors(10).w,10,10,[]),15,1))

%applyToSet(conf,detectors,lipImages(1:50),[],'stuff');


% applyToSet(conf,detectors,faces.train_faces(1:50),[],'stuff','override',true);

% x1 = fevalArrays(cat(4,lipImages{1:5}),@(y) col(fhog(im2single(y),4,9,.2,0)));

save ~/mircs/experiments/experiment_0012/lipImages_x x

% 





% load ~/mircs/experiments/experiment_0012/lipImages.mat
 load ~/mircs/experiments/experiment_0012/lipImages_x x

load ~/storage/misc/imageData_new;
%imageData = initImageData;
conf.get_full_image = false;
imageSet = imageData.train;
cur_t = imageSet.labels;
facesPath = fullfile('~/mircs/experiments/common/faces_cropped_new.mat');
load(facesPath);
%%

% extract all the lip images.

imageIDS = imageSet.imageIDs;
debug_info.ims = lipImages;
frameScale = 80;
subScale = 41;

% first, crop out the lip detections in all images. 

lip_bbs = getBBS(conf,imageSet.imageIDs,imageSet.lipBoxes,imageSet.faceBoxes,frameScale,subScale);

% get the bounding boxes of the images.

imageSet = imageData.train;

lip_bbs_orig = getBBS(imageSet.imageIDs,imageSet.lipBoxes,imageSet.faceBoxes,.5);
lip_bbs_orig = round(makeSquare(lip_bbs_orig));
lip_bbs_orig = inflatebbox(lip_bbs_orig,[1.3 1.3],'both',false);

sel_ = 1:size(lip_bbs_orig,1);;% > -.5 & imageSet.labels';
t_sel = imageSet.labels(sel_);
lipImages_orig = multiCrop(conf,imageSet.imageIDs(sel_),lip_bbs_orig(sel_,:));
save(fullfile(resDir,'lipImagesTrain.mat'),'lipImages_orig');
lipImages_64 = cellfun2(@(x) imResample(x,[64 64],'bilinear'),lipImages_orig);
qq = applyToSet(conf,detectors,lipImages_64,[],'stuff','override',true,'nDetsPerCluster',10);
save(fullfile(resDir,'lipImagesDetTrain.mat'),'newDets');


imageSet = imageData.test;

lip_bbs_orig = getBBS(imageSet.imageIDs,imageSet.lipBoxes,imageSet.faceBoxes,.5);
lip_bbs_orig = round(makeSquare(lip_bbs_orig));
lip_bbs_orig = inflatebbox(lip_bbs_orig,[1.3 1.3],'both',false);

sel_ = 1:size(lip_bbs_orig,1);;% > -.5 & imageSet.labels';
t_sel = imageSet.labels(sel_);
lipImages_orig = multiCrop(conf,imageSet.imageIDs(sel_),lip_bbs_orig(sel_,:));
save(fullfile(resDir,'lipImagesTest.mat'),'lipImages_orig');
lipImages_64 = cellfun2(@(x) imResample(x,[64 64],'bilinear'),lipImages_orig);
qq = applyToSet(conf,detectors,lipImages_64,[],'stuff_test','override',true,'nDetsPerCluster',10);
[newDets,dets,allScores] = combineDetections(qq);
save(fullfile(resDir,'lipImagesDetTest.mat'),'newDets');


% montage3(lipImages_orig(t_sel));
% 
%
% dets = det_union(qq);
% 
% showSorted(lipImages_64,newDets.cluster_locs(:,12),150);







windowSize = [40];
debug_info.ref_imgs = lipImages;
[new_bb,dists] = refineDetections(conf,imageSet.imageIDs,...
    lipImages_orig,lip_bbs_orig,x,windowSize);
   
lipScores = {};
lipIms = {};


find(~cellfun(@(x)size(x,1)==41 && size(x,2)==41,lipIms),1,'first')
lipIms(8)

x_lips = fevalArrays(cat(4,lipIms{:}),@(y) col(fhog(im2single(y),4)));
D = l2(x_lips',x');
scores = sum(exp(-D/10),2);
 
[s,is] = sort(scores+(imageSet.faceScores>-.7)','descend');
mImage(lipIms(is(1:5:500)));
plot(cumsum(cur_t(is)))


% [d,id] = sort(D,2,'ascend');
lipScores = -10^6*ones(1,4000);

lipIms = {};
for k = 1:4000%length(cur_t)
    k            
%     if (~(cur_t(k) || k < 1000))
%         continue;
%     end
%     break;
%     k = 813
   
    currentID = imageSet.imageIDs{k}
    [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
    bbox = round(imageSet.faceBoxes(k,1:4));
    bbox = clip_to_image(bbox,I);
    mouthBox = round(imageSet.lipBoxes(k,1:4)-bbox([1 2 1 2]));
    % mouthBox = round(inflatebbox(mouthBox,[3 1],'both',false));
    I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    bc = round(boxCenters(mouthBox));
    bb = (round(inflatebbox([bc bc],[40 40],'both',true)));
    
%     lipIms{k} = arrayCrop(I,[bb(2) bb(1) 1],[bb(4) bb(3) 3],0);
%     assert(size(lipIms{k},1)==41 && size(lipIms{k},2)==41);
%     continue;        
    %%mm = I(bb(2):bb(4),bb(1):bb(3),:);
    mm = arrayCrop(I,[bb(2) bb(1) 1],[bb(4) bb(3) 3],0);
%     xx = col(fhog(im2single(mm),4));    
%     D = l2(xx',x');
%     [d,id] = sort(D,2,'ascend');
    
    lipIms{end+1} = mm;
%     lipScores{end+1} = sum(exp(-d/10));
        
% %     clf; 
% %     subplot(1,2,1); imagesc(mm); axis image;
% %     knn = 16;
% %     subplot(1,2,2); montage2(cat(4,lipImages{id(1:knn)}),struct('hasChn',true));
% %     sum(exp(-d/10))
% %     pause
    
    
end


%     [d,id] = sort(D,2,'ascend');



pChns = chnsCompute();
pChns.pColor.enabled = 0;
d = cellfun2(@(x) chnsCompute(x,pChns),lipImages);
dd = cellfun2(@(x) col(cat(3,x.data{:})),d);
dd = cat(2,dd{:});

[mus,covs,ps] = vl_gmm(dd,3,'verbose');
mu_m = mus';
covs_m = reshape(covs,[1 size(covs)]);
ps_m = row(ps);
sharedCov = 0;

d = cellfun2(@(x) chnsCompute(x,pChns),lipIms);
dd = cellfun2(@(x) col(cat(3,x.data{:})),d);
dd_test = cat(2,dd{:});
% x is nXD mu is Dxk, Sigma is 1xDxk, PComponents is 1xk, SharedCov = 0, final is 1
% (for diagonal covariance matrix);
log_lh = gmm_posteriors(dd_test',mu_m, covs_m, ps_m, 0, 1);
m_lip_train = logsumexp(log_lh,2);
% [m,im] = max(log_lh,[],2);

showSorted(lipIms,m_lip_train,100);

% do the same for faces....
d = cellfun2(@(x) chnsCompute(imResample(x,[40 40],'bilinear'),pChns),ims);
dd_faces = cellfun2(@(x) col(cat(3,x.data{:})),d);
dd_faces = cat(2,dd_faces{:});
[mus,covs,ps] = vl_gmm(dd_faces,4,'verbose');
mu_f = mus';
covs_f = reshape(covs,[1 size(covs)]);
ps_f = row(ps);
sharedCov = 0;

% do it for train faces.


d = cellfun2(@(x) chnsCompute(imResample(x,[40 40],'bilinear'),pChns),faces.train_faces);
dd_faces_train = cellfun2(@(x) col(cat(3,x.data{:})),d);
dd_faces_train = cat(2,dd_faces_train{:});

log_lh = gmm_posteriors(dd_faces_train',mu_f, covs_f, ps_f, 0, 1);
m_faces = logsumexp(log_lh,2);
% [m,im] = max(log_lh,[],2);

showSorted(faces.train_faces,10000*(imageData.train.faceScores>-.5)'-m_lip_train,300);
vl_gmm
plot(m_faces); hold on; plot(m_lip_train,'r');


showSorted(faces.train_faces,m_faces,200);

[r,ir] = sort(m_faces,'descend');
mImage(faces.train_faces(ir(1:10:1000)));
% showSorted(lipImages,-m,100);

% log_lh = gmm_posteriors(x',mu, covs, ps, 0, 1);
% m = logsumexp(log_lh,2);
% [r,ir] = sort(m,'descend');
% mImage(lipImages(ir(1:30:end)));
% showSorted(lipImages,-m,100);

% plot(m,cur_t,'d')


[r,ir] = sort(m,'descend');
mImage(lipIms(ir(1:3:1000)));

pp1 = sum(pp,2);

% find the posterior

% mImage(lipIms);

dd = sum(exp(-D/10),2);

% try the CPMC release.

