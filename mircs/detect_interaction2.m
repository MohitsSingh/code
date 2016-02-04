sz = [40 40];
descs_train = getAllDescs(conf,model,train_faces,sz,'~/data/train_descs.mat');
descs_test = getAllDescs(conf,model,test_faces,sz, '~/data/test_descs.mat');

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
   
non_action_train = ismember(t_train_all,face_action_classes);
action_train = ismember(t_train_all,[conf.class_enum.DRINKING,...    
    conf.class_enum.BRUSHING_TEETH]);
    

% % non_action_faces = get_full_image(non_action_train);
% % action_faces = [train_faces_cigar,train_faces_cup,train_faces_brush,train_faces_phone];
% 


action_faces = train_faces(action_train);
non_action_faces = train_faces(non_action_train);

%mImage(action_faces,[1 5 inf]);

% mImage(action_faces,[1 10])

conf.features.vlfeat.cellsize = 8;
sz1 = [64 64];
[f0,sizes] = imageSetFeatures2(conf,action_faces,true,sz1);
f1 = imageSetFeatures2(conf,non_action_faces,true,sz1);

D = l2(f0',f1');

% show for each image the nearest neighbors.
[~,INN] = sort(D,2,'ascend');

bbb = nnDisplay(action_mouths,non_action_mouths,INN,5);

imwrite(cat(1,bbb{1:100}),'aaa_lips.jpg');

train_new_sals = showNeighborDifferences(f0,f1,action_faces,INN,sizes,[],non_action_faces);
test_new_sals = showNeighborDifferences(f0_test,f1,action_test_mouths,INN_test,sizes,[],non_action_mouths);



%%
[f0,sizes] = imageSetFeatures2(conf,lipImages_train,true,sz1);
f1 = imageSetFeatures2(conf,non_action_mouths,true,sz1);

D = l2(f0',f1');

% show for each image the nearest neighbors.
[~,INN] = sort(D,2,'ascend');

bbb = nnDisplay(lipImages_train,non_action_mouths,INN,5);

imwrite(cat(1,bbb{1:100}),'aaa_lips.jpg');

train_new_sals = showNeighborDifferences(f0,f1,lipImages_train,INN,sizes,[],non_action_mouths);
[f0_test,sizes] = imageSetFeatures2(conf,lipImages_test,true,sz1);
D_t = l2(f0_test',f1');
[~,INN_t] = sort(D_t,2,'ascend');

test_new_sals = showNeighborDifferences(f0_test,f1,lipImages_test,INN_t,sizes,[],non_action_mouths);

% [ws,b,sv,coeff] = train_classifier(f0,f1);


sz = [50 50];
descs_train_mouth = getAllDescs(conf,model,lipImages_train,sz,'~/data/train_descs_mouth.mat');
descs_test_mouth = getAllDescs(conf,model,lipImages_test,sz, '~/data/test_descs_mouth.mat');
model.numSpatialX = 1;
model.numSpatialY = 1;
feats_train_mouth = getBOWFeatures(conf,model,lipImages_train,train_new_sals,descs_train_mouth);
feats_test_mouth = getBOWFeatures(conf,model,lipImages_test,test_new_sals,descs_test_mouth);

%%
multiWrite(lipImages_train,'action_mouths_train');
multiWrite(lipImages_test,'action_mouths_test');
[sal_train,inds] = multiRead('action_mouths_train/res','.png');
[sal_test,inds] = multiRead('action_mouths_test/res','.png');

feats_train_mouth = getBOWFeatures(conf,model,lipImages_train,[],descs_train_mouth);
feats_test_mouth = getBOWFeatures(conf,model,lipImages_test,[],descs_test_mouth);

neg_train = feats_train_mouth(:,non_action_train);
pos_train = feats_train_mouth(:,action_train);

[ws,b] = train_classifier(pos_train,neg_train,.01);

scores = feats_test_mouth'*ws;
[r,ir] = sort(scores,'descend');
figure,imshow(multiImage(lipImages_test(ir(1:100)),false))

ws_drink=  train_classifier(feats_train_mouth(:,t_train),feats_train_mouth(:,~t_train));
ws_drink=  train_classifier(feats_train_mouth(:,t_train),neg_train);

scores2 = feats_test_mouth'*ws_drink;
%%
clf;
total_score = double(scores>.1)+1*scores2;
[q,iq] = sort(total_score,'descend');
clf,figure(1);imshow(multiImage(lipImages_test(iq(1:100)),false));

[prec,rec,aps] = calc_aps2(total_score,t_test,sum(test_labels));
figure(2);
plot(rec,prec); title(num2str(aps));
% 
% figure,imshow(multiImage(images))
% 
% U = zeros([size(images{1}) length(images)]);
% for k = 1:length(images)
%     U(:,:,k) = images{k};
% end
% 
% vl_imarraysc(U)
% 
% V= zeros([size(lipImages_train{1}) length(images)]);
% for k = 1:length(images)
%     V(:,:,:,k) = lipImages_train{action_train(k)};
% end
% 
% 
% m = multiImage(lipImages_train(action_train),false);
% imwrite(m,'actions.jpg');

%%

%% original params
lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,[80 80],'both','abs'));

lipImages_train = multiCrop(conf,train_faces,lipBoxes_train_r_2,[50 50]);

figure,imshow(multiImage(lipImages_train(t_train),false));

%lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[64 64],'both','abs'));
lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[80 80],'both','abs'));

lipImages_test = multiCrop(conf,test_faces,lipBoxes_test_r_2,[50 50]);
%% for searching...
lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,[40 40],'both','abs'));

lipImages_train_ = multiCrop(conf,train_faces,lipBoxes_train_r_2,[50 50]);

figure,imshow(multiImage(lipImages_train(t_train),false));

%lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[64 64],'both','abs'));
lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[80 80],'both','abs'));

lipImages_test_ = multiCrop(conf,test_faces,lipBoxes_test_r_2,[50 50]);

%%

aaa = lipImages_train(t_train);

sal_test_large = multiCrop(conf,sal_test,[],...
    2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);

straw_inds = [1 5 6 14 18 19 21 23 27 31 38 46 51];
straw_rects = selectSamples(conf,aaa(straw_inds),'straw_rects_train');
straw_rects = imrect2rect(straw_rects);
straw_imgs = multiCrop(conf,aaa(straw_inds),straw_rects,[40 40]);
conf.features.vlfeat.cellsize = 8;
feats_straw = imageSetFeatures2(conf,straw_imgs,true,[]);
conf.features.winsize = [5 5];
figure,imshow(showHOG(conf,mean(feats_straw,2).^2))
c = makeCluster(double(feats_straw),[]);
conf.clustering.num_hard_mining_iters = 12;
lipImages_train_large = multiCrop(conf,lipImages_train,[],...
    2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);
c_straw = train_patch_classifier(conf,c,lipImages_train_large(~t_train),'suffix','straw1','override',false);
figure,imshow(showHOG(conf,c_straw))
lipImages_test_large = multiCrop(conf,lipImages_test,[],...
    2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);

qq_straw = applyToSet(conf,c_straw,lipImages_test_large,[],'straw1_test','override',false);
[prec_straw,rec_straw,aps_straw] = calc_aps(qq_straw,t_test,sum(test_labels));

cup_inds = [4 9 10 11 14 16 26 30 37 44 49 50 60 62];
cup_rects = selectSamples(conf,aaa(cup_inds),'cup_rects_train');
cup_rects = imrect2rect(cup_rects);
cup_imgs = multiCrop(conf,aaa(cup_inds),cup_rects,[40 40]);
conf.features.vlfeat.cellsize = 8;
feats_cup = imageSetFeatures2(conf,cup_imgs,true,[]);
conf.features.winsize = [5 5];
figure,imshow(showHOG(conf,mean(feats_cup,2).^2))
c = makeCluster(double(feats_cup),[]);
conf.clustering.num_hard_mining_iters = 12;
lipImages_train_large = multiCrop(conf,lipImages_train,[],...
    2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);
c_cup = train_patch_classifier(conf,c,lipImages_train_large(~t_train),'suffix','cup1','override',false);
figure,imshow(showHOG(conf,c_cup))
% lipImages_test_large = multiCrop(conf,lipImages_test,[],...
%     2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);
qq_cup = applyToSet(conf,c_cup,lipImages_test_large,[],'cup1_test','override',false);
[prec_cup,rec_cup,aps_cup] = calc_aps(qq_cup,t_test,sum(test_labels));

plot(rec_cup,prec_cup)

bottle_inds = [8 13 15 25 34 35 45 55];
bottle_rects = selectSamples(conf,aaa(bottle_inds),'bottle_rects_train');
bottle_rects = imrect2rect(bottle_rects);
bottle_imgs = multiCrop(conf,aaa(bottle_inds),bottle_rects,[40 40]);
conf.features.vlfeat.cellsize = 8;
feats_bottle = imageSetFeatures2(conf,bottle_imgs,true,[]);
conf.features.winsize = [5 5];
figure,imshow(showHOG(conf,mean(feats_bottle,2).^2))
c = makeCluster(double(feats_bottle),[]);
conf.clustering.num_hard_mining_iters = 12;
lipImages_train_large = multiCrop(conf,lipImages_train,[],...
    2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);
c_bottle = train_patch_classifier(conf,c,lipImages_train_large(~t_train),'suffix','bottle1','override',true);
figure,imshow(showHOG(conf,c_bottle))
% lipImages_test_large = multiCrop(conf,lipImages_test,[],...
%     2*[size(lipImages_train{1},1) size(lipImages_train{1},2)]);
qq_bottle = applyToSet(conf,c_bottle,lipImages_test_large,[],'bottle1_test','override',false);
[prec_bottle,rec_bottle,aps_bottle] = calc_aps(qq_bottle,t_test,sum(test_labels));
plot(rec_bottle,prec_bottle)
figure,imshow(multiImage(lipImages_test_large(qq_bottle.cluster_locs(1:50,11))))


for k = 1:100
    c = test_faces{qq_bottle.cluster_locs(k,11)};
    subplot(1,2,1);imshow(c);
    E = edge(im2double(rgb2gray(c)),'canny');
    subplot(1,2,2);imshow(E);
    pause;
    
end

figure,imshow(multiImage(test_faces(qq_bottle.cluster_locs(1:50,11))))

% % 
% % aaa = lipImages_train(t_train);
% % aaa = multiCrop(conf,aaa,[],2*[size(aaa{1},1) size(aaa{1},2)]);
% % 
% % figure,imshow(multiImage(aaa,true))
% % figure,imshow(multiImage(sal_train(t_train),true))
% % 
% % f = cellfun(@(x) sum(x(:)), sal_train(t_train));
% % 
% % [r,ir] = sort(f,'descend');
% % 
% % figure,imshow(multiImage(aaa(ir),true))
% % 
% % 
% % detect_straws_cups2

% check quality of landmark detections
faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);
[f,sizes] = imageSetFeatures2(conf,train_faces,true,sz1);
D = l2(f',f');
[keep,faceData] = extractFaceData(conf,faceLandmarks_train_t,train_faces,~t_train,D);

non_action_train_t = non_action_train(goods);
action_train_t = action_train(goods);

action_feats = cat(2,[faceData(action_train_t).D]);
non_action_feats = cat(2,[faceData(non_action_train_t).D]);

train_faces_t = train_faces(keep);
mImage(train_faces_t(action_train_t))

%[ws,b] = train_classifier(action_feats,non_action_feats);
feats = double([action_feats non_action_feats]')/255;
y = -ones(size(feats,1),1);
y(1:length(action_faces)) = 1;
svm_model = svmtrain(y, feats,'-t 2 -c 1');

% faceLandmarks_test_t = faceLandmarks_test(test_face_scores>=min_test_score);



% test_feats = double(cat(2,[faceData_test.D]))'/255;
[~,~,ss] = svmpredict(zeros(size(test_feats,1),1),test_feats,svm_model);

% ss = ws'*double(test_feats)-b;

[r,ir] = sort(ss,'descend');
test_faces_t = test_faces(keep_test);
mImage(test_faces_t(ir(1:10:400)));

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);

goods_ = false(size(faceLandmarks_train_t));

for k = 1:length(faceLandmarks_train_t)
    goods(k) = size(faceLandmarks_train_t(k).xy,1) == 68;
end

train_faces_t = train_faces(goods);
faceLandmarks_train_t = faceLandmarks_train_t(goods);
for q = 1:length(faceLandmarks_train_t)
    faceLandmarks_train_t(q).xy = boxCenters(faceLandmarks_train_t(q).xy);
    faceLandmarks_train_t(q).xy = faceLandmarks_train_t(q).xy(:);
end

allBoxes = cat(2,[faceLandmarks_train_t.xy]);

t_train_t = t_train(goods);
figure,imshow(multiImage(train_faces_t(t_train_t)))

lipBoxes_train_r_t = lipBoxes_train_r(goods,:);

D = l2(allBoxes',allBoxes');
[~,inn] = sort(D,2,'ascend');

aaa = find(t_train_t);
%for k = 1:length(aaa)
for k = 11:length(aaa)
    k
    clf,imshow(train_faces_t{aaa(k)});
    hold on;
    plotBoxes2(lipBoxes_train_r_t(aaa(k),[2 1 4 3]));
    
    boxEstimates = mean(allBoxes(:,inn(k,1:15)),2);
    plot(boxEstimates(1:end/2),boxEstimates(end/2+1:end),'r+');
    for b = 1:size(boxEstimates,1)/2
        text(boxEstimates(b),boxEstimates(b+size(boxEstimates,1)/2),num2str(b),'color','y');
    end
    pause;
end

