%%%%%% Experiment 13 %%%%%%%
% Nov. 19, 2013 - like exp9, but using my own landmark localization/mouth
% detection
echo off;
if (~exist('toStart','var'))
    initpath;
    weightVector = [1 10 10 0*-.01 10 3 1 1 10 1 0];
    
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');
    addpath('/home/amirro/code/3rdparty/PCA_Saliency_CVPR2013 - v2');
    addpath('/home/amirro/code/3rdparty/guy');
    addpath('/home/amirro/code/3rdparty/ssdesc-cpp-1.1.1');
    config;
    %load imageData_new; % which image-data to load? the one by zhu, or my face detection + mouth detection?
    LL = load('imageData_new2_exp1_3.mat');
%     LL = load('imageData_new.mat');
    %imageData = initImageData;
    toStart = 1;
    conf.get_full_image = false;
    imageSet = LL.imageData.train;
    face_comp = [imageSet.poses.yaw]';
    cur_t = imageSet.labels;
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [8 8];
    conf.detection.params.detect_add_flip = 0;
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t);
    %facesPath = fullfile('/home/amirro/mircs/experiments/experiment_0008/faces.mat');
%     load '/home/amirro/mircs/experiments/experiment_0001/sals_new.mat';
%     L = load('/home/amirro/mircs/experiments/experiment_0001_improved/exp_result.mat');
    load(facesPath);
end
%%

conf.get_full_image = false;
% feats_train = cell(1,length(imageData.train.imageIDs));
% ir = 1:4000;
% ir = 1:5532;
for q =1:length(imageSet.imageIDs)
    k = ir(q);
    %     k
    q
            if (~cur_t(k))
%                 continue;
            end
%     if (imageSet.faceScores(k) < -.7)
%         continue;
%     end
    %     Rs_train{k} =
    %     k = ir(10) % 8,  10 , 11
    % 7: 2335line too straight (not convex), too long
    %k = 128 app
    feats_train{k} = extractCandidateFeatures5(conf,imageSet,[],k,true,score_fun,q);
end
if (length(feats_train) < length(imageSet.imageIDs))
    feats_train{length(imageSet.imageIDs)} = [];
end


% save ~/mircs/experiments/experiment_0008/feats_train.mat feats_train
%%
% 1: find cups : prominent regions covering the mouth entirely, and
% largely covering the imSubsResizeface, but not too overlapping with the face.
sel_ = 1:length(imageSet.imageIDs);
% sel_ = find(cur_t);

curFaces = faces.train_faces(sel_);
curFeats = feats_train(sel_);
comps_ = comps(sel_);
poses = posemap(comps_);
faceScores = imageSet.faceScores(sel_);
%%
scores = zeros(size(curFeats));
M = -1000;
score_fun = @segment_grading;
X = {};
Y = {};
for q = 1:length(scores)
    r = curFeats{q};
    if (isempty(r))
        scores(q) = M;
        continue;
    end
    s = score_fun(r);
    [scores(q),is_q] = max(s);
%     if (cur_t(q))
        r.ints_face = r.ints_face/r.face_area;
        r.ints_mouth = r.ints_mouth/r.mouth_area;
        r.areas = r.areas/prod(r.winSize);
        ints_not = r.ints_not./r.areas;
        if (cur_t(q))
            cur_sel = is_q;
        else
            cur_sel = 1:length(r.ints_face);
        end
        
        
        
        
% % %         X{end+1} = [r.areas(cur_sel);...
% % %         r.bboxes(cur_sel,:)';...
% % %         r.gradientStrengths(cur_sel);...
% % %         r.gradientStrengths_noface(cur_sel);...
% % %         r.ints_face(cur_sel);...
% % %         r.ints_mouth(cur_sel);...
% % %         ints_not(cur_sel);...
% % %         repmat(r.mouth_area/prod(r.winSize),1,length(cur_sel))
% % %         r.ucmStrengths(cur_sel);...
% % %         r.ucmStrengths_noface(cur_sel)];
% % %         
% % %         if (cur_t(q))
% % %             Y{end+1} = true(length(cur_sel),1);            
% % %         else
% % %             Y{end+1} = false(length(cur_sel),1);
% % %         end                                                   
end

% % % 
% % % X1 = cat(2,X{:})';
% % % Y1 = cat(1,Y{:});
% % % 
% % % B = TreeBagger(10,X1,Y1,'OOBPred','on');
% % % plot(oobError(B))
% % % 
% % % 
% % % xx = (B.predict(X1));
% % % preds = cellfun(@(x) x-0,xx)==49;
% % % 
% % % pred2 = zeros(size(scores));
% % % lengths = cellfun(@(x) size(x,2),X);
% % % ll = [0 cumsum(lengths)];
% % % for k = 1:length(ll)-1
% % %     pred2(k) = max(preds(ll(k)+1:ll(k+1)));
% % % end
% % % 


% % % showSorted(faces.train_faces,pred2,200);
% [r,ir] = sort(pred2,'descend');
% for k  
% label('number of grown trees')
% ylabel('out-of-bag classification error')
%scores(~(~isinf(poses) & abs(poses)>=15)) = M;
% scores(abs(poses) > 90) = M;

% scores(isinf(poses)) = M;
% tt = 45;
% scores(abs(poses)>tt) = M;
% scores = scores +(faceScores*.05)';


T_face = faceScores < -.5;
scores(T_face) = scores(T_face)+2*faceScores(T_face)';
t0 = faceScores<-.35;
t1 = faceScores<-.55;
t2 = faceScores<-.65;

% scores = -.4-scores


% 
% scores(t2)  = scores(t2)-2;
% scores(t1 & ~t2)  = scores(t1 & ~t2)-1;
% scores(t0 & ~t1) = scores(t0 & ~t1)-3;
% ttt = -.5;
% scores(faceScores<ttt) = scores(faceScores<ttt)-2;


%scores(faceScores<ttt)-1;
% rrr = (0*T_saliency.stds+T_saliency.means_inside-1*T_saliency.means_outside)';
% scores = scores+.1*(rrr(sel_)');
scores(isnan(scores)) = M;
%scores(scores < 900) = min(scores(scores~=M));
scores(scores < -900) = min(scores(scores > -900));
% scores = scores-faceScores;
[prec,rec,aps] = calc_aps2(scores',cur_t(sel_));
[r,ir] = sort(scores,'descend');
s = showSorted(curFaces,scores,150);
% imwrite(s,'~/mircs/experiments/experiment_0005/frontal_cups_rule.png');