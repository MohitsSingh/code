%%
% show lip detection results on some lip images...
%
totalMax = 0;
for k = 4000:length(newImageData)
    k
    curImageData = newImageData(k);
%     if (~curImageData.label),continue,end;
    L_bb = load(j2m('~/storage/s40_lip_detection_full/',curImageData));
%     I = getImage(conf,curImageData);

%     bb = {};
% %     for k = 1:length(L_bb.bbs)
%         if (size(L_bb.bbs{k},2)==6)
%             bb{end+1} = L_bb.bbs{k};
%         end
%     end
    bbs = cat(1,L_bb.bbs{:});
    
    
%     bbs = cat(1,L_bb.bbs{:});
    %bbs = bbs(bbs(:,5)>30,:);
%     [r,ir] = sort(bbs(:,5),'descend'); bbs = bbs(ir(1:min(size(bbs,1),5)),:);
%     bbs = rotate_bbs(bbs,I,bbs(:,end));
    
    score = -10;
    if (~isempty(bbs))
        
        bbs(abs(bbs(:,7))>0,:) = [];
%         bbs(bbs(:,5)<30,:) = [];
        score = max(bbs(:,5));
        if (totalMax < score)
            totalMax = score
        end
        if (score > 10)
            [r,ir] = sort(bbs(:,5),'descend');% bbs = bbs(ir(1:min(size(bbs,1),5)),:);
            bbs = bbs(ir(1),:)
            [M,landmarks,face_box,face_poly] = getSubImage(conf,curImageData,1,true);
            I = imResample(M,[100 100],'bilinear');
            bbs = rotate_bbs(bbs,I,bbs(:,end));
            clf; imagesc(I); axis image; hold on; plotPolygons(bbs(:),'g');drawnow;             
%             plotBoxes(bbs,'g');drawnow;
            title(num2str(score));
            pause;
        end
    end     
end

% initpath;
% config;
% 

%%
allObjTypes = {'cup','bottle','straw'};
conf.features.vlfeat.cellsize = 8;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
conf.features.winsize = [8 8];
conf.detection.params.detect_min_scale = .5;
allClusters_ = {};

allImagesEnlarged_ = {};

for iObjectType = 1   
    objTypes = (allObjTypes(iObjectType));
    imgDir  = ['/home/amirro/storage/data/drinking_extended/' objTypes{1}];
    resDir = ['/home/amirro/storage/data/drinking_extended/' objTypes{1} '/annotations'];
    mkdir(resDir);
        bbLabeler(objTypes,imgDir,resDir);
    Is = {};
    XX = {};
    d = dir(fullfile(imgDir,'*.jpg'));
    for k = 1:length(d)
        k
        fName = fullfile(resDir,[d(k).name '.txt']);
        [objs,bbs] = bbGt( 'bbLoad', fName);
        bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
        bb = bbs(1:4);
        %     end
        I = imread(fullfile(imgDir,d(k).name));
        bb = round(makeSquare(bb));
        bb = inflatebbox(bb,[2 2],'both',false);
        I_ = cropper(I,round(bb));clf;imagesc(I_); axis image; pause; continue;
        I_ = imresize(I_,[64 64],'bilinear');
        if (length(size(I_))==2)
            I_ = repmat(I_,[1 1 3]);
        end
        XX{end+1} = col(vl_hog(im2single(I_),conf.features.vlfeat.cellsize,'NumOrientations',9));
        Is{end+1} = I_;
                
%         XX{end+1} = col(vl_hog(im2single(flip_image(I_)),conf.features.vlfeat.cellsize,'NumOrientations',9));
%         Is{end+1} = flip_image(I_);
    end
%     montage2(cat(4,Is{:}),struct('hasChn',1))
    
    XX = cat(2,XX{:});
    [IDX,C] = kmeans2(XX',3,struct('nTrial',10));
    %       clusters = makeClusters(X,[]);
    [clusters,ims,imss] = makeClusterImages(Is,C',IDX',XX,['clusters_' objTypes{1}]);
%     for k = 1:length(imss)
%         curImages = alignByJittering(conf,imss{k},false);
%         x = imageSetFeatures2(conf,mat2cell2(curImages,[1 1 1 size(curImages,4)]),true,[64 64]);
%         allClusters_{end+1} = makeCluster(x,[]);
%     end
    
    allClusters_{iObjectType} = clusters;
end
%%
save allClusters_ allClusters_
allClusters_ = [allClusters_{:}];
% train against many "clean" faces
[images,inds] = multiRead(conf,'/home/amirro/storage/data/faces/all_cropped/','jpg',[],[],200);
conf.clustering.num_hard_mining_iters = 5;
conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_min_scale = .5;
clusters_trained = train_patch_classifier(conf,allClusters_,images,'suffix','drink_mircs','override',true,'C',.001);

% alignByDetection(


conf.detection.params.detect_add_flip = 1;
for k = 1:length(clusters_trained)
    k
    clf;imagesc(showHOG(conf,clusters_trained(k))); axis image;% title('before');
%     subplot(1,3,2);imagesc(showHOG(conf,clusters_trained_phase2(k))); axis image; title('after');        
%     subplot(1,3,3);imagesc(showHOG(conf,clusters_trained(k).w-clusters_trained_phase2(k).w)); axis image; title('diff');%     
    pause
end

I_sub_neg = all_I_subs(~imageSet.labels);
clusters_trained_phase2 = train_patch_classifier(conf,clusters_trained,[],'suffix','drink_mircs_1','override',false,'C',.001);

for k = 1:length(clusters_trained_phase2)
    k
    clf;subplot(1,3,1);imagesc(showHOG(conf,clusters_trained(k))); axis image; title('before');
    subplot(1,3,2);imagesc(showHOG(conf,clusters_trained_phase2(k))); axis image; title('after');        
    subplot(1,3,3);imagesc(showHOG(conf,clusters_trained(k).w-clusters_trained_phase2(k).w)); axis image; title('diff');
    
    pause
end
%  montage2(cat(4,I_sub_neg{1:100}),struct('hasChn',true));



%%
conf.get_full_image = false;
qq_train = applyToSet(conf,clusters_trained,train_ids(train_labels),[],'drink_mircs','override',true,'disp_model',true,...
    'uniqueImages',true,'nDetsPerCluster',10,'visualizeClusters',true);

%%

initpath;
config;
conf.class_subset = conf.class_enum.DRINKING;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

conf.max_image_size = inf;
% prepare the data...

%%
%%

load train_landmarks_full_face.mat;
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat

% remove the images where no faces were detected.
train_labels = train_labels(train_dets.cluster_locs(:,11));
test_labels = test_labels(test_dets.cluster_locs(:,11));
all_test_labels = all_test_labels(test_dets.cluster_locs(:,11));
all_train_labels = all_train_labels(train_dets.cluster_locs(:,11));

% and do the same for the ids.
%
[faceLandmarks_train,lipBoxes_train,faceBoxes_train] = landmarks2struct(train_landmarks_full_face);
train_face_scores = [faceLandmarks_train.s];
[r_train,ir_train] = sort(train_face_scores,'descend');
[faceLandmarks_test,lipBoxes_test,faceBoxes_test] = landmarks2struct(test_landmarks_full_face);
test_face_scores = [faceLandmarks_test.s];
[r_test,ir_test] = sort(test_face_scores ,'descend');

train_labels_sorted = train_labels(ir_train);
test_labels_sorted = test_labels(ir_test);
m_train = multiImage(train_faces(ir_train(train_labels_sorted)),true);
% figure,imshow(m_train);
m_test = multiImage(test_faces(ir_test(test_labels_sorted)),true);
% figure,imshow(m_test);
% find 'true' lip coordinates.
debug_ = 1;



%%

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);



%%
close all;
train_faces = train_faces(train_face_scores>=min_train_score);
get_full_image = get_full_image(train_face_scores>=min_train_score);
% train_faces_4 = train_faces_4(train_face_scores>=min_train_score);


test_faces = test_faces(test_face_scores>=min_test_score);
test_faces_2 = test_faces_2(test_face_scores>=min_test_score);
% test_faces_4 = test_faces_4(test_face_scores>=min_test_score);

lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);
faceLandmarks_test_t = faceLandmarks_test(test_face_scores>=min_test_score);

lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,[80 80],'both','abs'));

lipImages_train = multiCrop(conf,train_faces,lipBoxes_train_r_2,[50 50]);

lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[80 80],'both','abs'));

lipImages_test = multiCrop(conf,test_faces,lipBoxes_test_r_2,[50 50]);

%%
conf.suffix = 'rgb';
dict = learnBowDictionary(conf,train_faces,true);
model.numSpatialX = [2];
model.numSpatialY = [2];
model.kdtree = vl_kdtreebuild(dict) ;
model.quantizer = 'kdtree';
model.vocab = dict;
model.w = [] ;
model.b = [] ;
model.phowOpts = {'Color','RGB'};
% figure,imshow(getImage(conf,train_ids{train_dets.cluster_locs(550,11)}))
%%
% % for k = 1:length(t_train)
% %     %     if ~t_train(k)
% %     %         continue;
% %     %     end
% %     clf;
% %     imshow(get_full_image{k});
% %     hold on;
% %     bc = boxCenters(faceLandmarks_train_t(k).xy);
% %     bc = bc/2+32;
% %     plot(bc(:,1),bc(:,2),'r.');
% %
% %     %     plotBoxes2(faceLandmarks_train_t(k).xy);
% %     pause(.1);
% % end
%%
xy_train = {};
for k = 1:length(t_train)
    bc = boxCenters(faceLandmarks_train_t(k).xy);
    %     bc = bc/2+32;
    xy_train{k} = bc(:);
end

lengths_train = cellfun(@length,xy_train);

lipReader;
allDrinkingInds = [cupInds_1 cupInds_2 cupInds_3 strawInds bottleInds];
t_train = train_labels(train_face_scores>=min_train_score);
ff = find(t_train);
ff = ff(allDrinkingInds);
t_train = false(size(t_train));
t_train(ff) = true;
mImage(train_faces(t_train));

close all;

tt_train = lengths_train == 136;
t_train_tt = t_train(tt_train);
train_faces_tt = get_full_image(tt_train);
train_faces_tt_x = train_faces(tt_train);
faceLandmarks_train_tt = faceLandmarks_train_t(tt_train);
mImage(train_faces_tt(t_train_tt));

xy_train = cat(2,xy_train{tt_train});

drinkingRects = selectSamples(conf,train_faces(t_train),'drinkingRects_train');%drinkingRects_train
% drinkingRects = selectSamples2(conf,train_faces(t_train),'drinkingRects_train_poly');%drinkingRects_train
drinkingRects = imrect2rect(drinkingRects)';
drinkingRects = drinkingRects/2+32;
drinkingRects = inflatebbox(drinkingRects',1,'both',false)';
drinkingRects = drinkingRects(:,ismember(find(t_train),find(tt_train)));

displayRectsOnImages(drinkingRects',train_faces_tt(t_train_tt));


% d_clusters_trained = train_patch_classifier(conf,d_clusters,train_faces(~t_train),'suffix','drinkers','override',false);


% remove the drinking rects no longer included in the training set

% choose the nearest neighbors only from the drinking faces.

% xx_train = imageSetFeatures2(conf,train_faces_tt_x,true,[80 80]);

% dd_train = l2(xx_train',xx_train(:,t_train_tt)');
dd_train = l2(xy_train',xy_train(:,t_train_tt)');

sigma_ = 10000;
% sigma_ = 10;
b_train = exp(-dd_train/sigma_);
% b = b.*(1-eye(size(b)));
b_train = bsxfun(@rdivide,b_train,sum(b_train,2));
%%
debug_ = false;
for k = 1:length(t_train_tt)
    k
    if (debug_)
        if (~t_train_tt(k))
            continue;
        end
    end
    xy_current = xy_train(:,k);
    xy_estimated = xy_train(:,t_train_tt)*b_train(k,:)';
    if (debug_)
        clf;
        subplot(2,2,1);
        
        imshow(train_faces_tt{k});
        hold on;
        plot(32+.5*xy_current(1:end/2),32+.5*xy_current((end/2+1):end),'r.');
        plot(32+.5*xy_estimated(1:end/2),32+.5*xy_estimated((end/2+1):end),'g.');
    end
    %
    box_estimated = drinkingRects*b_train(k,:)';
    
    %
    %     figure,imagesc(R)
    if (debug_)
        plotBoxes2(box_estimated([2 1 4 3])','m','LineWidth',2);
        %         pause(.1);
        subplot(2,2,2);
        segments = vl_slic(single(vl_xyz2lab(vl_rgb2xyz(im2single(train_faces_tt{k})))), 20, .1) ;
        [segImage,c] = paintSeg(train_faces_tt{k},segments);
        imagesc(segImage); axis image;
        
        subplot(2,2,3);
        alpha_ = .5;
        R = drawBoxes(zeros(128),drinkingRects',b_train(k,:),2);
        imshow((alpha_*jettify(R)+(1-alpha_)*im2double(train_faces_tt{k})))
        
        pause;
        
        
        
    end
    
    % show the estimated mask...
    %
    %     % now estimate the location for drinking...
    boxesEstimated_train(k,:) = round(box_estimated);
    
    
end

%%
% boxesEstimated_train2 = boxesEstimated_train;
% boxesEstimated_train(:,end) = boxesEstimated_train(:,end) + 20;
% boxesEstimated_train(:,1) = boxesEstimated_train(:,1) - 10;
displayRectsOnImages(boxesEstimated_train(t_train_tt,:),train_faces_tt(t_train_tt));

%%

lipWindowSize = [60 60];
hogWindowSize = [60 60];
model2 = model;
lipImages_train_2 = multiCrop(conf,train_faces_tt,boxesEstimated_train,lipWindowSize);
mImage(lipImages_train_2(t_train_tt));
conf.features.vlfeat.cellsize = 8;
[feats_train,sz] = imageSetFeatures2(conf,lipImages_train_2(t_train_tt),true,[40 40]);
conf.features.winsize = sz{1};

[C,IC] = vl_kmeans(feats_train,5, 'NumRepetitions', 100);
d_clusters = makeClusterImages(lipImages_train_2(t_train_tt),C,IC,feats_train,...
    'drinking_appearance_clusters');
% conf.detection.params.detect_min_scale

d_clusters_trained = train_patch_classifier(conf,d_clusters,train_faces(~t_train),'suffix','drinkers','override',true);
conf.detection.params.detect_min_scale = .8;
d_clusters_trained_res_ = applyToSet(conf,d_clusters_trained,test_faces_tt(t_test_tt),[],'drinkers_res','override',true,...
    'rotations',0);

% boxesEstimated_test
d_clusters_trained_res = applyToSet(conf,d_clusters_trained,test_faces_tt,[],'drinkers_res2','override',true,...
    'rotations',0,'perImageMasks',Rs);

[prec,rec,aps] = calc_aps(d_clusters_trained_res,t_test_tt)
plot(rec,prec)


% Rs

% mImage(lipImages_train_2(1:50:end));

% lipImages_test_tt = lipImages_test(tt);
%
% mImage(lipImages_test_tt(t_test_tt));

model2 = model;
model2.numSpatialX = [2 4];
model2.numSpatialY = [2 4];

ppp_train = [];
% qunatized_train = quantizeFeatures(conf,model2,lipImages_train_2,[]);
all_descs_train = getAllDescs(conf,model,train_faces_tt,[],'lip_train_descs.mat');

ppp_train = [ppp_train;getBOWFeatures(conf,model2,train_faces_tt,boxesEstimated_train,all_descs_train)];
conf.features.vlfeat.cellsize = 8;
% ppp_train =[ ppp_train ;imageSetFeatures2(conf,lipImages_train_2,true,hogWindowSize)];

cur_y_train = 2*(t_train_tt==1)-1;
% cur_y_test = 2*(cur_test_labels==k)-1;
ss2 = ss;
ss2 = '-t 0 -c .01 w1 1';
cur_svm_model= svmtrain(cur_y_train, double(ppp_train'),ss2);

%%
%%
% for k = 1:length(t_test)
%     if ~t_test(k)
%         continue;
%     end
%     clf;
%     imshow(test_faces{k});
%     hold on;
%     bc = boxCenters(faceLandmarks_test_t(k).xy);
%     plot(bc(:,1),bc(:,2),'r.');
%     %     plotBoxes2(faceLandmarks_test_t(k).xy);
%     pause;
% end
%%
xy_test = {};
for k = 1:length(t_test)
    bc = boxCenters(faceLandmarks_test_t(k).xy);
    xy_test{k} = bc(:);
end

lengths_test = cellfun(@length,xy_test);

tt_test = lengths_test == 136;
t_test_tt = t_test(tt_test);
test_faces_tt = test_faces_2(tt_test);
test_faces_tt_x = test_faces(tt_test);
faceLandmarks_test_tt = faceLandmarks_test_t(tt_test);
mImage(test_faces_tt(t_test_tt));

xy_test = cat(2,xy_test{tt});

% choose the nearest neighbors only from the drinking faces.
% xx_test = imageSetFeatures2(conf,test_faces_tt_x,true,[80 80]);

dd_test = l2(xy_test',xy_train(:,t_train_tt)');
% dd_test = l2(xx_test',xx_train(:,t_train_tt)');

b_test = exp(-dd_test/sigma_);
% b_test = b_test.*(1-eye(size(b_test)));
b_test = bsxfun(@rdivide,b_test,sum(b_test,2));


%%
Rs = {};
debug_ = false;
for k = 1:length(t_test_tt)
    k
    if (debug_)
        if (~t_test_tt(k))
            continue;
        end
    end
    xy_current = xy_test(:,k);
    xy_estimated = xy_train(:,t_train_tt)*b_test(k,:)';
    if (debug_)
        clf;
        subplot(2,2,1);
        
        imshow(test_faces_tt{k});
        hold on;
        plot(32+.5*xy_current(1:end/2),32+.5*xy_current((end/2+1):end),'r.');
        plot(32+.5*xy_estimated(1:end/2),32+.5*xy_estimated((end/2+1):end),'g.');
    end
    %
    box_estimated = drinkingRects*b_test(k,:)';
    
    %
    %     figure,imagesc(R)
    R = drawBoxes(zeros(128),drinkingRects',b_test(k,:),2);
    Rs{k} = R;
    if (debug_)
        plotBoxes2(box_estimated([2 1 4 3])','m','LineWidth',2);
        %         pause(.1);
        subplot(2,2,2);
        segments = vl_slic(single(vl_xyz2lab(vl_rgb2xyz(im2single(test_faces_tt{k})))), 20, .1) ;
        [segImage,c] = paintSeg(test_faces_tt{k},segments);
        imagesc(segImage); axis image;
        
        subplot(2,2,3);
        alpha_ = .5;
        
        imshow((alpha_*jettify(R)+(1-alpha_)*im2double(test_faces_tt{k})))
        
        pause;
        
        
        
    end
    
    % show the estimated mask...
    %
    %     % now estimate the location for drinking...
    boxesEstimated_test(k,:) = round(box_estimated);
    
    
end

%%
% boxesEstimated_test(:,end) = boxesEstimated_test(:,end) + 20;
% boxesEstimated_test(:,1) = boxesEstimated_test(:,1) - 10;
displayRectsOnImages(boxesEstimated_test(t_test_tt,:),test_faces_tt(t_test_tt));

% displayRectsOnImages(boxesEstimated(t_test_tt,:),test_faces_tt(t_test_tt));
%%
%%
lipImages_test_2 = multiCrop(conf,test_faces_tt,boxesEstimated_test,lipWindowSize);
mImage(lipImages_test_2(t_test_tt));
lipImages_test_tt = lipImages_test(tt);

% mImage(lipImages_test_tt(t_test_tt));

all_descs_test = getAllDescs(conf,model,test_faces_tt,[],'lip_test_descs.mat');
ppp_test = [];
ppp_test = [ppp_test;getBOWFeatures(conf,model2,test_faces_tt,boxesEstimated_test,all_descs_test)];
% ppp_test =[ ppp_test ;imageSetFeatures2(conf,lipImages_test_2,true,hogWindowSize)];

w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;
cur_model_res = ppp_test'*w;

%sel_ = [cigarSetTest;cupSetTest;brushSetTest;phoneSetTest];
% sel_ = 1:length(cur_model_res);
%%
% cur_svm_model = svm_models{2};
% w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;
% cur_model_res = ppp_test'*w;%w'*test_samples;
cur_model_res_ = cur_model_res;%+ .0005*test_faces_scores_r(tt)';
[r,ir] = sort(cur_model_res_,'descend');

m = mImage(lipImages_test_2(ir(1:1:50)));
[prec,rec,aps] = calc_aps2(cur_model_res_,t_test_tt,sum(t_test))
%%
plot(rec,prec)


%%

face_action_classes = [conf.class_enum.DRINKING,...
    conf.class_enum.BLOWING_BUBBLES,...
    conf.class_enum.BRUSHING_TEETH,...
    conf.class_enum.SMOKING,...
    conf.class_enum.PHONING,...
    conf.class_enum.PLAYING_VIOLIN,...
    conf.class_enum.PLAYING_GUITAR,...
    conf.class_enum.APPLAUDING,...
    conf.class_enum.CLIMBING,...
    conf.class_enum.CLEANING_THE_FLOOR,...
    conf.class_enum.TAKING_PHOTOS,...
    conf.class_enum.LOOKING_THROUGH_A_MICROSCOPE,...
    conf.class_enum.LOOKING_THROUGH_A_TELESCOPE];

non_action_train = ~ismember(t_train_all,face_action_classes);
non_action_train = non_action_train(tt_train);
nonAction_boxes = boxesEstimated_train(non_action_train,:);
% displayRectsOnImages(nonAction_boxes(1:20:end,:),non_action_faces(1:20:end));

mImage(non_action_faces(1:20:end));
hold on;


%vl_imarray(cat(4,lipImages_train_2{non_action_train}))
non_action_lipImages = lipImages_train_2(non_action_train);
% non_action_faces = train_faces_tt(non_action_train);
% %
% % xx = imageSetFeatures2(conf,train_faces_tt,true,[]);
% % xx_non = imageSetFeatures2(conf,non_action_lipImages,true,[]);
conf.features.vlfeat.cellsize = 8;
nf.features.winsize = [8 8]*8/conf.features.vlfeat.cellsize;
xx = imageSetFeatures2(conf,lipImages_train_2,true,[]);
xx_non = imageSetFeatures2(conf,non_action_lipImages,true,[]);
% r = l2(xx',xx_non');
% r = l2(ppp_train',ppp_train');

non_action_lipImages_r = {};
for k = 1:length(non_action_lipImages)
    non_action_lipImages_r{k} = im2double(non_action_lipImages{k}(:));
end
non_action_lipImages_r = cat(2,non_action_lipImages_r{:});

%%
sigma_2 = 1;
b_train = exp(-r/sigma_2);
% b = b.*(1-eye(size(b)));
b_train = bsxfun(@rdivide,b_train,sum(b_train,2));
%%
debug_ = true;
for k = 1:length(lipImages_train_2)
    k
    if (debug_)
        if (~t_train_tt(k))
            continue;
        end
    end
    
    curIm = lipImages_train_2{k};
    weights = b_train(k,:);
    %     subset = [1:k-1 k+1 length(lipImages_train_2)];
    %     weights = weights/sum(weights);
    q = reshape(non_action_lipImages_r*weights',size(curIm));
    
    if (debug_)
        clf;
        subplot(2,2,1);
        imshow(curIm);
        subplot(2,2,2);
        imshow(q);
        pause(.1);
        
    end
    
end

%%

xx_test = imageSetFeatures2(conf,lipImages_test_2,true,[]);

% r_test = l2(xx_test',xx_non');

% lipImages_train_2_r = {};
% for k = 1:length(lipImages_train_2)
%     lipImages_train_2_r{k} = im2double(lipImages_train_2{k}(:));
% end
% lipImages_train_2_r = cat(2,lipImages_train_2_r{:});
%
%%
% sigma_2 = 10;
% b_test = exp(-r_test/sigma_2);
% % b = b.*(1-eye(size(b)));
% b_test = bsxfun(@rdivide,b_test,sum(b_test,2));


%%
dd_test = l2(xy_test',xy_train(:,non_action_train)');
% dd_test = l2(xx_test',xx_train(:,t_train_tt)');
sigma_ = 5000;
b_test = exp(-dd_test/sigma_);
% b_test = b_test.*(1-eye(size(b_test)));
b_test = bsxfun(@rdivide,b_test,sum(b_test,2));


%%

close all;
debug_ = true;
means = {};
diffs = {};

%%
for k = 1:length(lipImages_test_2)
    k
    if (debug_)
        if (~t_test_tt(k))
            continue;
        end
    end
    
    curIm = lipImages_test_2{k};
    
    weights = b_test(k,:);
    
    %     [r,ir] = sort(weights,'descend');
    %     knn = 50;
    %     weights(ir(1:knn)) = 1/knn;
    %     weights(ir(knn+1:end)) = 0;
    
    %     subset = [1:k-1 k+1 length(lipImages_train_2)];
    %     weights = weights/sum(weights);
    q = reshape(non_action_lipImages_r*weights',size(curIm));
    
    %     subset = [1:k-1 k+1 length(lipImages_test_2)];
    %     weights = weights(subset);
    %     weights = weights/sum(weights);
    [c,ic] = sort(weights,'descend');
    
    q = reshape(q,size(curIm));
    
    x_q = imageSetFeatures2(conf,{q},true,[]);
    
    x = xx_non*weights';
    
    %     a = bsxfun(@minus,xx_non,x).^2;
    %     a = a*weights';
    
    x_diff = xx_test(:,k)-x;
    
    means{k} = x;
    diffs{k} = x_diff;
    
    rows = 3;
    cols = 3;
    if (debug_)
        clf;
        subplot(rows,cols,1);
        imshow(curIm);
        subplot(rows,cols,2);
        imshow(q);title('mean neighbors');
        subplot(rows,cols,3);
        imshow(multiImage(non_action_lipImages(ic(1:5)),false,true));
        title('neighbors');
        subplot(rows,cols,4);
        
        n = length(x)/31;
        s = round(sqrt(n));
        x = reshape(x,s,s,[]);
        conf.features.winsize = [s s];
        imshow(showHOG(conf,x.^5)); title('mean hog');
        
        subplot(rows,cols,5);
        x_diff = reshape(x_diff,s,s,[]);
        imshow(showHOG(conf,x_diff.^5)); title('hog diff');
        
        subplot(rows,cols,6);
        x_var = reshape(a,s,s,[]);
        imshow(showHOG(conf,x_var.^5)); title('hog var');
        
        subplot(rows,cols,7);
        imshow(showHOG(conf,x_diff./(x_var.^.5+eps))); title('hog diff  / var');
        
        %         subplot(rows,cols,8);
        %         x_q = reshape(x_q,s,s,[]);
        %         imshow(showHOG(conf,x_q.^2)); axis image; title('hog of mean');
        
        subplot(rows,cols,8);
        imagesc(sum(x_diff.^2,3).^.5); axis image; title('diff mag');
        
        %imshow(multiImage(non_action_lipImages(ic(1:5)),false,true))
        %         pause(.1);
        pause;
    end
end

m = cat(2,means{t_test_tt});
m = mean(m,2);

imshow(showHOG(conf,m)); title('mean of means');

m_diff = cat(2,diffs{t_test_tt});
m_diff  = mean(m_diff ,2);
imshow(showHOG(conf,m_diff)); title('mean of diffs');

figure,imagesc(sum(reshape(m_diff,s,s,[]),3)); axis image



%% shape of boundary, color of boundary, regularity.



%% do spatial transformations
for k = 1:length(lipImages_test_2)
    k
    if (debug_)
        if (~t_test_tt(k))
            continue;
        end
    end
    %     break;
    curIm = lipImages_test_2{k};
    
    weights = b_test(k,:);
    
    %     [r,ir] = sort(weights,'descend');
    %     knn = 50;
    %     weights(ir(1:knn)) = 1/knn;
    %     weights(ir(knn+1:end)) = 0;
    
    %     subset = [1:k-1 k+1 length(lipImages_train_2)];
    %     weights = weights/sum(weights);
    q = reshape(non_action_lipImages_r*weights',size(curIm));
    
    %     subset = [1:k-1 k+1 length(lipImages_test_2)];
    %     weights = weights(subset);
    %     weights = weights/sum(weights);
    [c,ic] = sort(weights,'descend');
    %     pts_train = pts_train(tt_train);
    
    p1 = pts_test{k};
    p2 = pts_train_noaction(ic);
    p1 = 32+p1/2;
    tforms = {};
    
    newImgs = {};
    newX = {};
    for ip = 1:50;%length(p2)
        %                 ip
        p2{ip} = 32+pts_train_noaction{ic(ip)}/2;
        Tform = cp2tform(p1,p2{ip},'similarity');
        pts = rect2pts(nonAction_boxes(ic(ip),:));
        
        ttt = non_action_faces{ic(ip)};
        
        newImgs{ip}= imtransform(ttt,fliptform(Tform),'XData',[1 size(ttt,2)],'YData',[1 size(ttt,2)]);
        
        [X,Y] = tforminv(Tform,pts(:,1),pts(:,2));
        newImgs(ip) = multiCrop(conf,newImgs(ip),nonAction_boxes(ic(ip),:),lipWindowSize);
        %         newImgs(ip) = non_action_lipImages(ic(ip));
        
        %         figure,imshow(a{1}); title('warped');
        % %         figure,imshow(curIm)
        % %         figure,imshow(non_action_lipImages{ic(1)});title('orig');;
        % %         figure,imshow(newImgs{ip});
        %
        %         hold on;
        %         plot(X,Y,'r-');
        %         figure,imshow(ttt);
        %         hold on;
        %         plot(pts(:,1),pts(:,2),'r-');
    end
    
    
    q = mean(im2double(cat(4,newImgs{:})),4);
    
    % %
    %     q = reshape(q,size(curIm));
    
    x_q = imageSetFeatures2(conf,newImgs,true,[]);
    x = mean(x_q,2);
    %     x = xx_non*weights';
    %     xx_t
    %     a = bsxfun(@minus,x_q,x).^2;
    %     a = mean(a,2);
    %
    x_diff = xx_test(:,k)-x;
    
    means{k} = x;
    diffs{k} = x_diff;
    
    rows = 3;
    cols = 3;
    if (debug_)
        clf;
        subplot(rows,cols,1);
        imshow(curIm);
        subplot(rows,cols,2);
        imshow(q);title('mean neighbors');
        subplot(rows,cols,3);
        imshow(multiImage(non_action_lipImages(ic(1:5)),false,true));
        title('neighbors');
        subplot(rows,cols,4);
        
        n = length(x)/31;
        s = round(sqrt(n));
        x = reshape(x,s,s,[]);
        conf.features.winsize = [s s];
        imshow(showHOG(conf,x.^5)); title('mean hog');
        
        subplot(rows,cols,5);
        x_diff = reshape(x_diff,s,s,[]);
        imshow(showHOG(conf,x_diff.^5)); title('hog diff');
        
        subplot(rows,cols,6);
        x_var = reshape(a,s,s,[]);
        imshow(showHOG(conf,x_var.^5)); title('hog var');
        
        %         subplot(rows,cols,7);
        %         imshow(showHOG(conf,x_diff./(x_var.^.5+eps))); title('hog diff  / var');
        %
        %         subplot(rows,cols,8);
        %         x_q = reshape(x_q,s,s,[]);
        %         imshow(showHOG(conf,x_q.^2)); axis image; title('hog of mean');
        
        subplot(rows,cols,8);
        imagesc(sum(x_diff.^2,3).^.5); axis image; title('diff mag');
        
        %imshow(multiImage(non_action_lipImages(ic(1:5)),false,true))
        %         pause(.1);
        pause;
    end
end

m = cat(2,means{t_test_tt});
m = mean(m,2);

imshow(showHOG(conf,m)); title('mean of means');

m_diff = cat(2,diffs{t_test_tt});
m_diff  = mean(m_diff ,2);
imshow(showHOG(conf,m_diff)); title('mean of diffs');

figure,imagesc(sum(reshape(m_diff,s,s,[]),3)); axis image


%%


% conf.features.vlfeat.cellsize
n
train_faces_tt = train_faces(tt_train);
test_faces_tt = test_faces(tt_test);
non_action_faces = train_faces_tt(non_action_train);
[xx] = imageSetFeatures2(conf,test_faces,true,[80 80]);
xx_non = imageSetFeatures2(conf,non_action_faces,true,[80 80]);
%
% xx = reshape(xx,8,8,31,[]);
% xx_non = reshape(xx_non,8,8,31,[]);

dd_test = l2(xy_test',xy_train(:,non_action_train)');
sigma_ = 5000;
b_test = exp(-dd_test/sigma_);
% b_test = b_test.*(1-eye(size(b_test)));
b_test = bsxfun(@rdivide,b_test,sum(b_test,2));
%%
for k = 1:length(lipImages_train_2)
    if (~t_test_tt(k))
        continue;
    end
    curXX = xx(:,k);
    curDiff = bsxfun(@minus,curXX,xx_non);
    %r = curDiff*b_test(k,:)';
    r = mean(curDiff,2);
    subplot(3,3,1);imagesc(test_faces_tt{k}); axis image;
    conf.features.winsize = [10 10];
    subplot(3,3,2); imshow(showHOG(conf,r).^.5);
    [v,iv] =sort(b_test(k,:),'descend');
    subplot(3,3,3);imshow(multiImage(non_action_faces(iv(1:5)),false,false));
    pause;
end

%% show some spagglom options

conf.get_full_image = false;
baseDir = '~/storage/s40_sp_agglom_2_partial/';
d = dir(fullfile(baseDir,'*.mat'));
%%
conf.get_full_image = false;
for k = 1:1:length(d)
    clc
    k
    I = getImage(conf,strrep(d(k).name,'.mat','.jpg'));
    clf; 
    R = load(fullfile(baseDir,d(k).name));
    
    Z = zeros(size2(I));
    for t = 1:length(R.res)
        Z = Z+imResample(normalise(R.res(t).fg_map),size2(I),'bilinear');
    end
    
    Z = normalise(Z);
    Z = addBorder(Z,3,0);
    subplot(2,2,1); imagesc2(I);
%     R = normalise(R.res);
    subplot(2,2,2); imagesc2(Z);
    pause;continue;
    subplot(2,2,3); %displayRegions(I,Z>.5);
    gc_segResult = getSegments_graphCut_2(imResample(I,.25,'bilinear'),imResample(Z,.25,'bilinear'),[],1);
    subplot(2,2,4); displayRegions(I,imResample(gc_segResult,size2(I),'nearest'));
    drawnow;
    pause;
end

%% show some fra_db stuff


load fra_db;
%sel_ = [fra_db.classID]~=5 & [fra_db.isTrain] & [fra_db.classID]==conf.class_enum.BRUSHING_TEETH;
sel_ = ~[fra_db.isTrain];
fra_db = fra_db(sel_);
%%
d = defaultPipelineParams();
%d.roiParams.absScale=-1;
m = 3;
n = m*m;
u = vl_colsubset(1:length(fra_db),n,'random');
% u =  [21    31    35   101   112   113   144   268   352];
clf;
%roiParams.infScale = 2;
roiParams.infScale = 4;
roiParams.absScale = -1;%200;
roiParams.centerOnMouth = 1;
%
figure(1)
for t = 1:length(u)
    k = u(t);
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k),roiParams);
    vl_tightsubplot(m,m,t);
    imshow(I);hold on;
%     vl_tightsubplot(m,m,2*(t-1)+2);
%     imagesc2(I);
    plotBoxes(rois(1).bbox,'g-','LineWidth',2);
    plotBoxes(rois(2).bbox,'m-','LineWidth',2);
    plotPolygons(rois(3).poly,'r-','LineWidth',2);    
    [kp_preds,goods] = loadKeypointsGroundTruth(fra_db(k),d.requiredKeypoints);
    confidences = kp_preds(:,3);
    kp_preds = kp_preds(:,1:2);
    kp_preds = kp_preds-repmat(roiBox(1:2),size(kp_preds,1),1);
    kp_preds = kp_preds*scaleFactor;
    
    plotPolygons(kp_preds,'b.','LineWidth',2,'MarkerSize',10)
%     dpc
end