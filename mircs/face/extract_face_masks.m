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
% load newFaceData4.mat;
% remove the images where no faces were detected.
train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));
all_test_labels=all_test_labels(test_dets.cluster_locs(:,11));
all_train_labels=all_train_labels(train_dets.cluster_locs(:,11));

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
t_train_all_tt = t_train_all(tt_train);
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
drinkingRects = inflatebbox(drinkingRects',1.5,'both',false)';
drinkingRects = drinkingRects(:,ismember(find(t_train),find(tt_train)));

displayRectsOnImages(drinkingRects',train_faces_tt(t_train_tt));

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
lipImages_train_2 = multiCrop(conf,train_faces_tt,boxesEstimated_train,lipWindowSize);
mImage(lipImages_train_2(t_train_tt));

%%
%%
xy_test = {};
for k = 1:length(t_test)
    bc = boxCenters(faceLandmarks_test_t(k).xy);
    xy_test{k} = bc(:);
end

lengths_test = cellfun(@length,xy_test);

tt_test = lengths_test == 136;
t_test_tt = t_test(tt_test );
t_test_all_tt = t_test_all(tt_test);
test_faces_tt = test_faces_2(tt_test );
test_faces_tt_x = test_faces(tt_test );
faceLandmarks_test_tt = faceLandmarks_test_t(tt_test );
mImage(test_faces_tt(t_test_tt));

xy_test = cat(2,xy_test{tt_test });

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
displayRectsOnImages(boxesEstimated_test(t_test_tt,:),test_faces_tt(t_test_tt));
%%
%%
lipImages_test_2 = multiCrop(conf,test_faces_tt,boxesEstimated_test,lipWindowSize);
mImage(lipImages_test_2(t_test_tt));
lipImages_test_tt = lipImages_test(tt);

multiWrite(train_faces_tt,'train_faces_tt');
multiWrite(test_faces_tt,'test_faces_tt');
[sal_train,inds] = multiRead('train_faces_tt/res','.png');
[sal_test,inds] = multiRead('test_faces_tt/res','.png');

faceMasks_train = getFaceMasks(train_faces_tt(t_train_tt));


