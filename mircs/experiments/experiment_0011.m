%%%%%% Experiment 11 %%%%%%%
% Nov. 16, 2013

% This is similar to experiment 0009, but I am using 
% my own landmark estimation, which is a result of experiment_8
% (detection of faces using dpm + keypoint transfer from aflw).


echo off;
if (~exist('toStart','var'))
    initpath;    
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');        
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');

    config;% extract the lip images from the detected
    load imageData_new2; 
%     imageData = imageData_new2;
%     clear imageData_new2;
    toStart = 1;
    conf.get_full_image = false;
    imageSet = imageData.train;    
    cur_t = imageSet.labels;
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [8 8];
    conf.detection.params.detect_add_flip = 0;
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t);    
    poses = 180*[imageData.train.poses.pitch]/pi;    
    for k = 1:size(imageSet.faceBoxes)
        curMouthBox = imageSet.lipBoxes(k,:);
        curFaceBox = imageSet.faceBoxes(k,:);
        curMouthBox = curMouthBox*(curFaceBox(3)-curFaceBox(1))/80 + ...
        curFaceBox([1 2 1 2]);
%         clf; imagesc(getImage(conf,imageSet.imageIDs{k})); hold on;
%         hold on;plotBoxes2(curFaceBox([2 1 4 3]),'r');
%         hold on;plotBoxes2(curMouthBox([2 1 4 3]),'g');
%         pause
        imageSet.lipBoxes(k,:) = curMouthBox;
    end    
end

%%
conf.get_full_image = false;
feats_train = cell(1,length(imageData.train.imageIDs));
% ir = 1:4000;
for q =1:length(imageData.train.imageIDs)
    k = ir(q)
%     q
        if (~cur_t(k))
%             continue;
        end
    if (imageData.train.faceScores(k) < -.6)
        continue;
    end
    %     Rs_train{k} =
%     k = ir(10) % 8,  10 , 11
    % 7: 2335line too straight (not convex), too long
    %k = 128 app
%     if (isempty(feats_train{k}))
% k = 128
        feats_train{k} = extractCandidateFeatures5(conf,imageSet,[],k,true,scorefun,q);
%     end
end

% feats_train{4000} = [];
%%
% 1: find cups : prominent regions covering the mouth entirely, and
% largely covering the imSubsResizeface, but not too overlapping with the face.


sel_ = 1:length(imageData.train.imageIDs);
% sel_ = find(cur_t);

curFaces = imgs_train(sel_);
curFeats = feats_train(sel_);
comps_ = poses(sel_);
faceScores = imageSet.faceScores(sel_);
%%
scores = zeros(size(curFeats));
M = -1000;
scorefun = @(r) -r.ints_face...
        +r.ints_mouth... % occlude much of mouth
        +(r.ints_face<.3)+...  % occlude face only partially
        ((r.bboxes(:,3)-r.bboxes(:,1))' < .7)+...
        ((r.bboxes(:,4)-r.bboxes(:,2))' < .7)+...
        (r.bboxes(:,2)' > 0.2)+...
        +0*(r.bboxes(:,4)' >= .3) +...
        (r.ucmStrengths)*.5+...
        0*(r.areas < .4);
for q = 1:length(scores)
    r = curFeats{q};
    if (isempty(r))
        scores(q) = M;
        continue;
    end
   
  
    s = scorefun(r);
    scores(q) = max(s);
end
%scores(~(~isinf(poses) & abs(poses)>=15)) = M;
% scores(abs(poses) > 90) = M;

scores(isinf(poses)) = M;
% tt = 20;
% scores(abs(comps_)>=tt) = M;
scores = scores-abs(comps_)*.01;
ttt = -.6;
scores = scores-(faceScores*.01)';
% scores(faceScores<ttt) = M;%scores(faceScores<ttt)-1;
% scores(faceScores>.3) = M;%scores(faceScores<ttt)-1;
% rrr = (0*T_saliency.stds+T_saliency.means_inside-1.5*T_saliency.means_outside)';
% scores = scores+ 0*.5*(rrr(sel_)');
scores(isnan(scores)) = M;
scores(scores == M) = min(scores(scores~=M));

% scores = scores-faceScores;
[prec,rec,aps] = calc_aps2(scores',cur_t(sel_));
[r,ir] = sort(scores,'descend');
s = showSorted(curFaces,scores,150);
% imwrite(s,'~/mircs/experiments/experiment_0005/frontal_cups_rule.png');

%% 2 : side cups : find strong, convex curves
conf.get_full_image = false;
feats_train2 = cell(1,length(imageData.train.imageIDs));
%%
[s,is] = sort(scores,'descend');
for q =1:length(imageData.train.imageIDs)
    q
    k = is(q);
% for k = 4000:-1:1
%     k=900
  
        if (~cur_t(k))
            continue;
        end
    if (imageData.train.faceScores(k) < -.6)
        continue;
    end
    if (~isinf(poses(k)) && poses(k) > 15)        %     Rs_train{k} =
%         feats_train2{k} = 
        extractCandidateFeatures3(conf,imageData.train,sal_train,k,true,true);
    end
end
%%
scores = zeros(size(curFeats));
M = -1000;
for q = 1:length(scores)
    r = feats_train2{q};        
    if (isempty(r))
        scores(q) = M;
        continue;
    end
        
    ucmStrengths = cat(1,r.ucmStrength);
    isConvex = cat(2,r.isConvex)';
    horz_extent = cat(1,r.horz_extent);

   	s = ucmStrengths+(horz_extent < .5);
%     s(~isConvex) = M;
        
    scores(q) = max(s);
end
%scores(~(~isinf(poses) & abs(poses)>=15)) = M;
% scores(abs(poses) > 90) = M;

scores(isinf(poses)) = M;
% tt = 60;
% scores(abs(poses)>tt) = M;
scores(faceScores<-.6) = M;
scores(scores == M) = min(scores(scores~=M));

% scores = scores+faceScores*.05;
[prec,rec,aps] = calc_aps2(scores',cur_t(sel_));
s = showSorted(curFaces,scores,70);