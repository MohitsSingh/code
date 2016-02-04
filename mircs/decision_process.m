echo off;
if (~exist('toStart','var'))
    initpath;
    weightVector = [1 10 10 0*-.01 10 3 1 1 10 1 0];
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');
    addpath('/home/amirro/code/3rdparty/PCA_Saliency_CVPR2013 - v2');
    addpath('/home/amirro/code/3rdparty/guy');
    config;
    imageData = initImageData;
    toStart = 1;
    conf.get_full_image = true;
    imageSet = imageData.train;
    face_comp = [imageSet.faceLandmarks.c]';

    cur_t = imageSet.labels;
    allScores = -inf*ones(size(cur_t));
    frs = {};
    pss = {};
    hand_scores = {};
    f = find(cur_t);
    strawInds_ = f([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54]); % for train only!!
    strawInds = strawInds_;
    Zs = {};
    fhog1 = @(x) fhog(im2single(x),4,9,.2,0);
    
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [8 8];
    conf.detection.params.detect_add_flip = 0;
    
    % get the ground truth for cups...
    %     [groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    %     gtParts = {groundTruth.name};
    %     isObj = cellfun(@any,strfind(gtParts,objectName));
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t)
    
    
    facesPath = fullfile('~/mircs/experiments/common/faces_cropped.mat');
    load(facesPath);
%     clusters_trained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','drink_mircs','override',false,'C',.001);

end
% should improve the face alignment...

%%

%%
close all;
debug_ = false;

% ff = find(cur_t);
% 505, the Obama image
% for k = ff([1 5 6 14 18 19 21 23 27 31 38 42 46 51 54])'

% for q = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ] % 526
% iv = [556 476 492 496 497 506 526  519 525 540 546 505 549 551 552 ];
% iv = 1:length(cur_t)
% iv = strawInds_;
% for q = 1:length(cur_t)
% debug_ = false;

% Rs = {};


%%
conf.get_full_image = true;
% qq_train = applyToSet(conf,clusters_trained,train_ids(train_labels),[],'drink_mircs','override',true,'disp_model',true,...
%     'uniqueImages',true,'nDetsPerCluster',10,'visualizeClusters',true);



%%

% all_I_subs = {};
conf.detection.params.detect_add_flip = 0;
conf.detection.params.detect_min_scale = .5;
conf.get_full_image = true;
allObjTypes = {'cup','bottle','straw'};

for q = 1489:length(cur_t)
    q
    k = iv(q)
    currentID = imageSet.imageIDs{k};
    if (~cur_t(k))
%                 continue;
    end
    I = getImage(conf,currentID);
    
    faceBoxShifted = imageSet.faceBoxes(k,:);
    lipRectShifted = imageSet.lipBoxes(k,:);
    faceLandmarks = imageSet.faceLandmarks(k);
    
    facePts = faceLandmarks.xy;
    facePts = boxCenters(facePts);
    
    box_c = round(boxCenters(lipRectShifted));
    sz = faceBoxShifted(3:4)-faceBoxShifted(1:2);
    
    
    %for scale = [1.5]%:.1:2]
    for scale =1.5
    
    bbox = round(inflatebbox([box_c box_c],floor(sz/scale),'both',true));
   
    
    % EXPERIMENTAL
%     bbox = faceBoxShifted;
%     bbox = round(inflatebbox([box_c box_c],sz*1.3,'both',true));
    bbox = clip_to_image(bbox,I);
    % EXPERIMENTAL
    I_sub_color = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub_color = imresize(I_sub_color,[80 NaN],'bilinear');
    I_sub_color = max(0,min(1,I_sub_color));
    
    all_I_subs{k} = I_sub_color;
    continue
    I_sub = rgb2gray(I_sub_color);
    
%     I_sub_color = images{q};
        
    clf; subplot(1,3,1);imagesc(I_sub_color); axis image;
    conf.clustering.min_cluster_size = 0;
    currentDets = applyToSet(conf,clusters_trained,{I_sub_color},[],'''','override',true,'disp_model',true,...
        'uniqueImages',true,'nDetsPerCluster',10,'visualizeClusters',false,'toSave',false);
        
    bb = cat(1,currentDets.cluster_locs);
    bb(:,1:4) = round(clip_to_image(bb(:,1:4),I_sub_color));
    
    H = computeHeatMap(I_sub_color,bb(:,[1:4 12]),'max');
%     Rs{k} = extractCandidateFeatures(conf,currentID,...
%         imageSet.faceBoxes(k,:),...
%         imageSet.lipBoxes(k,:),...
%         imageSet.faceLandmarks(k),debug_);
%      subplot(2,1,6); imagesc(H);
    subplot(1,3,2); imagesc(H); colormap jet; axis image; colorbar
    title(num2str(max(bb(:,12))));
    
    [rr,irr] = max(bb(:,12));
    subplot(1,3,3); imagesc(showHOG(conf,clusters_trained(irr))); axis image; 
    irr
    title(allObjTypes{ceil(irr/3)});
    pause
    
    end
%     [A,AA] = visualizeClusters(conf,{I_sub_color},currentDets,'add_border',...
%         false,'nDetsPerCluster',...
%         5,'gt_labels',labels,...
%         'disp_model',true,'height',64);
    if (debug_)
%         pause
    end
end
%%
% conf.detection.params.detect_min_scale = .3;
conf.detection.params.detect_add_flip = 0;
conf.detection.params.detect_min_scale = .5;
currentDets = applyToSet(conf,clusters_trained,all_I_subs,[],'face_mirc_det','override',true,'disp_model',true,...
        'uniqueImages',true,'nDetsPerCluster',10,'visualizeClusters',false,'toSave',true);
    

    
    %%
[newDets,dets,allScores] = combineDetections(currentDets);





%[ws,b,sv,coeff,svm_model] = train_classifier(allScores,neg_samples,C,w1,s);

% plot(rec,prec)      
% sss = newDets.cluster_locs(:,12);

sss = allScores;

% sss2 = zeros(size(allScores,1)/length(scale),size(allScores,2));
% 
% for k = 1:size(sss2,1)
%     sss2(k,:) = max(sss( (k-1)*6+1 : k*6,:));
% end
% 
% sss = sss2;

sss = normalise(sss);
sss = max(sss,[],2);
% sss = sum(sss,2);
% sss = sss;%1*ismember(face_comp,6:11);%+0*(imageSet.faceScores > -.85)';%+.00001*rand(size(sss));
% sss = min(sss,[],2);
% sss = mean(sss,2);
sss = sss;% +.5*rand(size(sss));
[prec,rec] = calc_aps2(sss,imageSet.labels);

[s,iv] = sort(sss,'descend');

% showSorted(all_I_subs,newDets.cluster_locs(:,12),100);


% showSorted(all_I_subs,sss,100);
% 
%%

debug_ = true;
for q = 1:length(cur_t)
    q
    k = iv(q);
    currentID = imageSet.imageIDs{k};
    if (~cur_t(k))
%         continue;
    end
    Rs{k} = extractCandidateFeatures(conf,currentID,...
        imageSet.faceBoxes(k,:),...
        imageSet.lipBoxes(k,:),...
        imageSet.faceLandmarks(k),debug_); 
    if (debug_) 
        pause 
    end
end

% save Rs Rs
%
%% 1. find all locations where the line is rather horizontal and spans a significant part of the image (cup1).

% 1 R = [contourLengths/size(E,1);...
%  2   ucmStrengths;...
%  3   row(pMean(:,1))/size(E,1);...
%  4   verticality';...
%  5   skinTransition;...
%  6   insides;...
%  7   nIntersections;...
%  8   areas./nnz(face_mask);...
%  9   inLips;...
%  10   horzDist';...
%  11  areas;...
%  12  bboxes'/size(E,1)];
%%
myScores = zeros(length(Rs),1);
            %1  2   3   4   5   6   7   8   9   10  11  12  13  14  15   
  weights = [1  5   0   0   20  1   0  0   10   10   0   0   0   0   0];

for k = 1:length(Rs)
    
    R = Rs{k};
    if (isempty(R))
        myScores(k) = -10;
        continue;
    end
    contourLengths = R(1,:);    
    areas = R(8,:);
    inLips = R(9,:);
    ucmStrengths = R(2,:);
    t1 = contourLengths>.3;
    horzDist = R(10,:);
%     t2 = areas>.1 & areas < .2;
    t2 = horzDist;
%     curScores = t1+uecmStrengths+1*t2+inLips;
    
    
    curScores = R'*weights';
    
%     curScores(isnan(curScores)) = -100;
        
    myScores(k) = max(curScores);    
end
myScores(isnan(myScores)) = -1000;
[v,iv] = sort(myScores,'descend');

L = load('~/mircs/experiments/experiment_0001/exp_result.mat');
%%
theScores = .03*myScores;
theScores = theScores + (L.train_saliency.stds+L.train_saliency.means_inside-L.train_saliency.means_outside)';
theScores = theScores+-0*ismember(face_comp,6:11)+(imageSet.faceScores > -.7)';
[prec,rec,aps,T] = calc_aps2(theScores,cur_t);
%%
% bbb = imageSet.faceBoxes;
% sz1 = (bbb(:,3)-bbb(:,1));
% 
% theScores = theScores+1000*(sz1>12);

M = showSorted(faces.train_faces,theScores,50);
imwrite(M,