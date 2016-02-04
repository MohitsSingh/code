%%  This script is an attempt to learn how to detect when a specific body-part,
% e.g, face, is interacting with something. The idea is to extract features
% which are both adaptive (relative to other image locations) and
% independent (local) and to build a classifier. 

%% first, construct a dataset of interacting vs. non-interacting faces.
prepareFaceData;
close all;

min_train_score = -.8;
train_faces_r = train_faces(train_face_scores>=min_train_score);
test_faces_r = test_faces(test_face_scores>=min_train_score);

mkdir('tmp_train');
mkdir('tmp_test');
multiWrite(train_faces_r,'tmp_try');

[occluded,occ_inds] = multiRead('tmp_try/occluded','.tif');
[not_occluded,not_occ_inds] = multiRead('tmp_try/not_occluded','.tif');
occ_inds = cat(1,occ_inds{:});
not_occ_inds = cat(1,not_occ_inds{:});

train_faces_re = multiCrop(conf,train_faces,round(inflatebbox(faceBoxes_train,1.5)),[128 128]);
train_faces_re_r = train_faces_re(train_face_scores>=min_train_score);

figure,imshow(train_faces_r{1})
figure,imshow(train_faces_re_r{1})

test_faces_re = multiCrop(conf,test_faces,round(faceBoxes_test),[128 128]);
test_faces_re_r = test_faces_re(test_face_scores>=min_train_score);

occluded = train_faces_re_r(occ_inds);
not_occluded = train_faces_re_r(not_occ_inds);

figure,imshow(multiImage(occluded(1:50:end)));
figure,imshow(multiImage(not_occluded(1:50:end)));
% figure,imshow(multiimage(train_faces_re(1:50)));title('after');
% figure,imshow(multiimage(train_faces(1:50)));title('before');
%%
conf.features.vlfeat.cellsize = 8;
[pos_feats,sz] = imageSetFeatures2(conf,occluded,true,[40 40]);
[neg_feats,sz] = imageSetFeatures2(conf,not_occluded,true,[40 40]);

[w,b] = train_classifier(pos_feats,neg_feats);
size(w)
test_feats = imageSetFeatures2(conf,test_faces_re_r,true,[40 40]);
conf.features.winsize =sz{1}(1:2);
figure,imshow(showHOG(conf,w))

test_res = test_feats'*w;
[r,ir] = sort(test_res,'descend');
% test_faces_r_small = multiCrop(conf,test_faces_re_r,[],[40 40]);
figure,imshow(multiImage(test_faces_re_r(ir(1:150)),false));

test_m = bsxfun(@times, test_feats,w);
%%
for k = 1:100
c = ir(k);
t = test_m(:,c);
t = reshape(t,[ conf.features.winsize 31]);
a = imresize((showHOG(conf,t)),[128 128]);
clf;
subplot(2,2,1);
imshow(test_faces_re_r{ir(k)});
subplot(2,2,2);
imshow(a);
t_ = sum(abs(t),3);
b = imresize(t_,[128 128]);

subplot(2,2,3);
imshow(jettify(b));


b = repmat(b,[1,1,3]);
b = b/max(b(:));
subplot(2,2,4);
imshow(b.*im2single(test_faces_re_r{ir(k)}));
pause;
end



%%
figure,imshow(multiImage(occluded(1:20)))
figure,imshow(multiImage(not_occluded(1:20)));
%%


%%

sz = sz{1};
%%
feats_pos_reshaped = {};

for k = 1:length(occluded)
    feats_pos_reshaped{k}= reshape(pos_feats(:,k),[],31);    
end
feats_neg_reshaped = {};
for k = 1:length(not_occluded)
    feats_neg_reshaped{k}= reshape(neg_feats(:,k),[],31);
end
nCells = prod(sz(1:2));
all_mats_pos = cat(1,feats_pos_reshaped{:});
all_mats_neg = cat(1,feats_neg_reshaped{:});
t_pos = cell(1,nCells);
t_neg = cell(1,nCells);
for k = 1:nCells
    t_pos{k} = all_mats_pos(k:nCells:end,:);
    t_neg{k} = all_mats_neg(k:nCells:end,:);
end

dist_mats = cell(1,nCells);
for k = 1:nCells
    dist_mats{k} = l2(t_pos{k},t_neg{k});
end


% sel_ = setdiff(1:nCells,9);
 sel_ = 1:nCells;
dists = cat(3,dist_mats{sel_});
dists_ = sum(dists,3);

nn_dists = cell(1,nCells);
for k = 1:nCells
    %[nn_cells(k).d,nn_cells(k).id] = sort(dist_mats{k},2,'ascend');
    nn_dists{k}= sort(dist_mats{k},2,'ascend');
end

nn_dists_ = cat(3,nn_dists{:});

[d,id] = sort(dists_,2,'ascend');

%%

%%

% 1. Show for each positive image the nearest negative images.
% Then, for each cell, show the sum to the distances of it's ten nearest 
% absolute neighbors. 

% 
% sal = {};
% 
% parfor k = 1:length(occluded)
%     k
%     img = occluded{k};
% figure,imshow(img)
% sal{k}= gbvs( img );
% end

addpath('/home/amirro/code/3rdparty/signatureSal');


%   paramRGB.colorChannels = 'rgb';

%   rgbMap = signatureSal( img , paramRGB );
    paramRGB = default_signature_param;
  paramRGB.mapWidth = 128;
  paramRGB.blurSigma = .05;
sal2 = {};
parfor k = 1:length(occluded)
    k
    img = occluded{k};
% figure,imshow(img)
    sal2{k}= signatureSal( img  , paramRGB );
end


sal_not = {};
parfor k = 1:length(not_occluded)
    k
    img = not_occluded{k};
% figure,imshow(img)
    sal_not{k}= signatureSal( img  , paramRGB );
end
% A thought - I can find a correlation between 
% visual saliency and body parts for a given action.

for k = 130:length(occluded)
    k
    img = occluded{k};
%     show_imgnmap(img,sal{k});
    curSal = sal2{k};
    curSal = repmat(curSal,[1 1 3]);
    
    imshow([img im2uint8(curSal)]);
    pause
% figure,imshow(img)
	
end
%     figure,imshow(out.master_map_resized)
% figure,imshow(img);

%%
close all;
knn = 20;
for k = 1:1:size(d,1)
    k
        q = occluded(k);
        nn = not_occluded(id(k,1:knn));
        q{1} = addBorder(q{1},3,[0 255 0]);
        a = reshape(sum(nn_dists_(k,1:knn,:),2),sz(1:2));
        imshow(multiImage([ q jettify(a) nn],false,false));
        pause
end
%%
% 2. Cycle over all cells. For each cell, find the nearest neighbors for the
% set of cells not including this one. Then show the distances induced over
% this cell.
%%
knn = 20;

m = cell(1,length(occluded));
for k = 1:length(m)
    m{k} = zeros(sz(1:2));
end
for k = 1:nCells
    k
%      partialDists(:,:,k) = sum(dists(:,:,setdiff(1:nCells,k)),3);

    partialDists =   sum(dists(:,:,setdiff(1:nCells,k)),3);     
    [d,id] = sort(partialDists,2,'ascend');
                
    for kk = 1:size(partialDists,1)
        m{kk}(k) = sum(dists(kk,id(kk,1:5),k));
    end
    
    % now re-compute the partial dists, disregarding the salient areas, per
    % image.
    
%     id(:,1:5)
end
%%
sel_2 = {};
for k = 1:length(m)
    sel_2{k} = find(m{k} > median(m{k}(:)))';
end
sel_2 = cat(1,sel_2{:});
figure,imagesc(sel_2)

m2 = {};
knn2 = 5;
for k = 1:length(m)
    partialDists = sum(dists(:,:,setdiff(1:nCells,sel_2(k,:))),3);
    [d,id2] = sort(partialDists,2);
    % show the nn....
    q = occluded(k);
    nn = not_occluded(id2(k,1:knn2));
        q{1} = addBorder(q{1},3,[0 255 0]);
%         a = reshape(sum(nn_dists_(k,1:knn,:),2),sz(1:2));
        nn2_image = (multiImage([q nn],false,true));
        nn1_image= multiImage([ q not_occluded(id(k,1:knn2))],false,true);
            
        imshow([nn1_image;nn2_image]);
        pause;
end

%%
for k = 1:length(m)
    m_ = imresize(m{k},[128 128]);
    m_ = m_/max(m_(:));
        
    
    mm = jettify(m_);
    imshow([occluded{k} im2uint8(mm) ...
        im2uint8(bsxfun(@times,m_.^2,im2double(occluded{k})))]);
      pause;  
end
%%
% imshow(multiImage(jettify(m(1:10)),false))
% choose 
% need partial nearest neighbors.....

% see if the rest of the face matches but one part doen't match. 
% this is an anomaly. 

% or, most of the face matches but one part is hard to reconstruct from 
% the remaining parts... 






% show reconstruction error...

H = neg_feats;
% H_ = pinv(H);
% Y = test_feats;
% q = H_*Y;
% rec = H*q;
% rec_err = sum((rec-Y).^2);
% [z,iz] = sort(rec_err,'ascend');
% 
% figure,imshow(multiImage(test_faces_re_r(iz(1:50))));
% 
% for k = 1:100
% c = iz(k);
% t = rec(:,c);
% t = reshape(t,[ conf.features.winsize 31]);
% a = imresize((showHOG(conf,t)),[128 128]);
% clf;
% subplot(2,2,1);
% imshow(occluded{ir(k)});
% subplot(2,2,2);
% imshow(a);
% t_ = sum(abs(t),3);
% b = imresize(t_,[128 128]);
% 
% subplot(2,2,3);
% imshow(jettify(b));
% 
% 
% b = repmat(b,[1,1,3]);
% b = b/max(b(:));
% subplot(2,2,4);
% imshow(b.*im2single(occluded{iz(k)}));
% pause;
% end

%%
mkdir('train_faces_true');

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
    
non_action_train = find(~ismember(t_train_all,face_action_classes));

train_faces_cigar = train_faces(cigarSetTrain(cigarTrain_w));
train_faces_cup = train_faces(cupSetTrain(cupTrain_w));
train_faces_brush = train_faces(brushSetTrain(brushTrain_w));
train_faces_blow = train_faces(blowSetTrain(blowTrain_w));
train_faces_phone = train_faces(phoneSetTrain(phoneTrain_w));

non_action_faces = get_full_image(non_action_train);
action_faces = [train_faces_cigar,train_faces_cup,train_faces_brush,train_faces_phone];

sz1 = [80 80];
[f0,sizes] = imageSetFeatures2(conf,action_faces,true,sz1);
f1 = imageSetFeatures2(conf,non_action_faces,true,sz1);

[ws,b,sv,coeff] = train_classifier(f0,f1);

figure,imagesc(HOGpicture(reshape(ws,sizes{1})))

f2 = imageSetFeatures2(conf,test_faces_2,true,sz1);

f2 = normalize_vec(f2);

q = ws'*f2;
[r,ir] = sort(q(1:1000),'descend');
test_faces__ = test_faces(1:1000);
figure,imshow(multiImage(test_faces__(ir(1:100))))

szz= sizes{1};
for k = 1:500
%     f2_1 = reshape(f2(:,k),szz);
    subplot(2,2,1);
    imagesc(test_faces__{ir(k)});axis image;
    subplot(2,2,2);
    
    imagesc(HOGpicture((reshape(ws.*f2(:,ir(k)),szz)))); axis image;title('weighted');
    subplot(2,2,3);
    imagesc(HOGpicture(reshape(f2(:,ir(k)),szz))); axis image;title('features');
    subplot(2,2,4);
    imagesc(HOGpicture(reshape(ws,szz))); axis image;title('w');
    pause;
end

D = l2(f0',f1');

% show for each image the nearest neighbors.
[~,INN] = sort(D,2,'ascend');
bbb = nnDisplay(action_faces,non_action_faces,INN,5);

imwrite(cat(1,bbb{1:100}),'aaa.jpg');

sigma_ = 10;
figure,imagesc(exp(-D/sigma_));colorbar

v = sum(exp(-D/sigma_),2);

[r,ir] = sort(v,'descend');
figure,imshow(multiImage(action_faces(ir(1:100))))


diffs = zeros(size(f0));
%%

non_action_faces = train_faces(non_action_train);
action_faces = [train_faces_cigar,train_faces_cup,train_faces_brush,train_faces_phone];
sz1 = [64 64];
[f0,sizes] = imageSetFeatures2(conf,action_faces,true,sz1);
f1 = imageSetFeatures2(conf,non_action_faces,true,sz1);

D = l2(f0',f1');

% show for each image the nearest neighbors.
[~,INN] = sort(D,2,'ascend');
bbb = nnDisplay(action_faces,non_action_faces,INN,5);

imwrite(cat(1,bbb{1:100}),'aaa.jpg');

%%

train_new_sals = showNeighborDifferences(f0,f1,action_faces,INN,sizes,[],non_action_faces);
D2 = l2(f1',f1');
% show for each image the nearest neighbors.
[~,INN2] = sort(D2,2,'ascend');
train_new_sals_neg = showNeighborDifferences(f1,f1,non_action_faces,INN2,sizes,[],non_action_faces);

model2.numSpatialX = 1;
model2.numSpatialY = 1;

pos_feats = getBOWFeatures(conf,model2,action_faces,train_new_sals);
neg_feats = getBOWFeatures(conf,model2,non_action_faces,train_new_sals_neg);

[ws,b,sv,coeff] = train_classifier(pos_feats,neg_feats);


f0_test = imageSetFeatures2(conf,test_faces,true,sz1);
D_test = l2(f0_test',f1');
% show for each image the nearest neighbors.
[~,INN_test] = sort(D_test,2,'ascend');
test_new_sals= showNeighborDifferences(f0_test,f1,test_faces,INN_test,sizes,[]);

test_feats = getBOWFeatures(conf,model2,test_faces,test_new_sals);

scores = ws'*test_feats;
[r,ir] = sort(scores,'descend');
imwrite(multiImage(test_faces(ir)),'actions.jpg');
figure,imshow(multiImage(test_faces(ir(1:100))));

t = false(size(t_test));
t(cigarSetTest) = true;
t(cupSetTest) = true;
t(blowSetTest) = true;
t(brushSetTest) = true;

figure,plot(cumsum(t(ir)))

[prec,rec,aps] = calc_aps2(scores(:),t);
plot(rec,prec);

feats_drink = pos_feats(:,49:(49+60));
feats_notdrink = pos_feats(:,[1:48 110:end]);

[ws2] = train_classifier(feats_drink,feats_notdrink);

scores2 = ws2'*test_feats;
[r2,ir2] = sort(scores2,'descend');
figure,imshow(multiImage(test_faces(ir2(1:100))));

[prec,rec,aps] = calc_aps2(scores2(:),t_test);
plot(rec,prec);

scores3 = scores2+.5*scores;
[prec,rec,aps] = calc_aps2(scores3(:),t_test);
plot(rec,prec);title(num2str(aps));

pos_feats = getBOWFeatures(conf,model2,action_faces,[]);
neg_feats = getBOWFeatures(conf,model2,non_action_faces,[]);

[ws,b,sv,coeff] = train_classifier(pos_feats,neg_feats);

test_feats = getBOWFeatures(conf,model2,test_faces,[]);

scores = ws'*test_feats;
[r,ir] = sort(scores,'descend');
figure,imshow(multiImage(test_faces(ir(1:100))));



vars = var(f1,0,2);
vars = reshape(vars,[10 10 31]);
figure,imagesc(sum(vars,3))



theSals = showNeighborDifferences(f0_test(:,t_test),f1,test_faces(t_test),INN_test(t_test,:),sizes,test_sal(t_test));

theSals = showNeighborDifferences(f0_test,f1,test_faces,INN_test,sizes,test_sal);



%% Now, learn a general appearance model, and see how results change when incorporating 
% the saliency measure.
[f_cups,f_sz] = imageSetFeatures2(conf,train_cup_images(1:10),true,[48 48]);
cc = makeCluster(double(f_cups),[]);
conf.features.winsize = f_sz{1};
cc_trained = train_patch_classifier(conf,cc,test_faces(~t_train),'suffix','cups1','override',true);
figure,imshow(showHOG(conf,cc));
conf.detection.params.detect_max_windows_per_exemplar = inf;
qq = applyToSet(conf,cc_trained,test_faces,[],'cup_1','override',false,'uniqueImages',true,'sals',theSals);
[prec,rec,aps] = calc_aps(qq,t_test);
qq.cluster_locs(:,12)

figure,imshow(multiImage(jettify(theSals(qq.cluster_locs(1:100,11)))))
figure,imshow(multiImage((test_faces(qq.cluster_locs(1:100,11)))))

vv = visualizeLocs2_new(conf,test_faces,qq.cluster_locs);
figure,imshow(multiImage(vv(1:100)));



