%%detect_interaction_3
initpath;
config;

conf.class_subset = conf.class_enum.DRINKING;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

conf.max_image_size = inf;
% prepare the data...


%%

load train_landmarks_full_face.mat;
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat

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

locs_train = train_dets.cluster_locs;
% locs_train_t = locs_train(locs_train(:,12)>=min_train_score,:);

find_train = find(train_labels);
% train_ids_d = train_ids(locs_train(:,11));

% figure,imshow(getImage(conf,train_ids{locs_train_t(find_train(1),11)}))

%%

%%


% get a random subset of the train faces and use logistic regression to
% score the faces..
imshow(multiImage(train_faces(ir_train(1:20:end))));
sss = r_train(1:20:end);
ttt = false(size(sss));
ttt([1:56 58:63 66:67 69 72 74 76 80 96:98 113 127 131 147:148 157 164 172 177]) = true;

[w_,b_] = logReg(sss(:),ttt(:));

% curScores = sigmoid(sss*w+ b);
% plot(curScores);
% hold on;
% plot(ttt,'r+');
%

% figure,plot(sigmoid(r_train*w_+b_))

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);

% missingTestInds = (test_face_scores < min_train_score) & test_labels';
% test_ids_d_1 = test_ids(test_dets.cluster_locs(:,11));
% missingTestRects = selectSamples(conf,test_ids_d_1(missingTestInds),'missingTestTrueFaces');
% missingTestRects = cat(1,missingTestRects{:});
% missingTestRects(:,4) = missingTestRects(:,4)+missingTestRects(:,2);
% missingTestRects(:,3) = missingTestRects(:,3)+missingTestRects(:,1);
% save missingTestRects.mat missingTestRects missingTestInds

%%
close all;
train_faces = train_faces(train_face_scores>=min_train_score);
get_full_image = get_full_image(train_face_scores>=min_train_score);

test_faces = test_faces(test_face_scores>=min_test_score);
test_faces_2 = test_faces_2(test_face_scores>=min_test_score);

lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

%%
for k = 385:length(train_faces)    
    if (~t_train(k))
        continue;
    end
    clf; imshow(train_faces{k});
    hold on;
    plotBoxes2(lipBoxes_train_r(k,[2 1 4 3]),'Color','g','LineWidth',2);
    pause;
end


% learn.... 



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
% figure,imshow(getImage(conf,train_ids{train_dets.cluster_locs(550,11)}));

% segment the faces...
%%

f = find(t_train);
% f = 1:length(t_train);

%%
getTheFaceMasks = 0;
if (getTheFaceMasks)
% matlabpool
getFaceMasks(get_full_image);

faceMasks_train = getFaceMasks(get_full_image,ff_mean);

for k = 1:length(faceMasks_train)
    faceMasks_train{k} = faceMasks_train{k} > 0;
end

faceMasks_test = getFaceMasks(test_faces_2,ff_mean);

for k = 1:length(faceMasks_test)
    faceMasks_test{k} = faceMasks_test{k} > 0;
end

% save ff_mean ff_mean;
save faceMasks.mat faceMasks_train faceMasks_test;

sel_ = find(t_train);
end
% sel_ = 1:length(t_train);
%%


train_masks_2 = smoothMasks(faceMasks_train);
test_masks_2 = smoothMasks(faceMasks_test);

r = train_masks_2{1};

f = find(r);


q  = bwdist(1-r);

clear m_train m_test
train_feats = getBOWFeatures(conf,model,get_full_image,train_masks_2,[]);
test_feats = getBOWFeatures(conf,model,test_faces_2,test_masks_2,[]);

save maskedFeats.mat train_feats test_feats;

y_train = 2*(t_train==1)-1;
svm_model = svmtrain(y_train,double(train_feats'),'-t 0 -c 1');

[~,~,ss] = svmpredict(zeros(size(test_feats,2),1),double(test_feats'),svm_model);

[r,ir] = sort(ss,'ascend');
mImage(test_faces(ir(1:50)));
%%

% sel_ = find(t_train_all==conf.class_enum.DRINKING);

rr_train = {};

rr = getLogPolarRegions([128 128],6,[],15,[]);
sel_ = 1:length(train_faces);
nSectors = 8;
for k = 1:length(sel_)
    
    kk = sel_(k);
    kk
%     imagesc(faceMasks_train{kk})
    curMask = faceMasks_train{kk};
    I = get_full_image{kk};
    rr_train{k} = getLogPolarRegions([128 128],nSectors,[],15,curMask,I);
end

sel_ = 1:length(test_faces);
rr_test = {};
for k = 1:length(sel_)
    
    kk = sel_(k);
    kk
%     imagesc(faceMasks_train{kk})
    curMask = faceMasks_test{kk};
    I =test_faces_2{kk};
    rr_test{k} = getLogPolarRegions([128 128],nSectors,[],15,curMask,I);
end

conf.features.vlfeat.cellsize = 8;



descs_train = getAllDescs(conf,model,get_full_image,[],'~/data/train2_descs.mat');
descs_test = getAllDescs(conf,model,test_faces_2,[], '~/data/test2_descs.mat');

descs_train = getSingleBowFeatures(get_full_image,model,descs_train);
descs_test = getSingleBowFeatures(test_faces_2,model,descs_test);

model.numSpatialX = [1 2];
model.numSpatialY =[1 2];
maskedBowTrain = getMaskedBow(get_full_image,model,descs_train,rr_train);
maskedBowTest = getMaskedBow(test_faces_2,model,descs_test,rr_test);

maskedBowTrain = getMaskedHOG(conf,get_full_image,rr_train);
maskedBowTest = getMaskedHOG(conf,test_faces_2,rr_test);
maskedBowTrain = maskedBowTrain';
maskedBowTest = maskedBowTest';

t_train = t_train_all == conf.class_enum.DRINKING;

f = find(t_train);
f_ = find(~t_train);

% maskedBowTrain_learn = maskedBowTrain(:,1:2:end);
% maskedBowTrain_concept = maskedBowTrain(:,2:2:end);

concept_inds = [f(1:2:end);f_(1:3:end)];
learn_inds = setdiff(1:length(t_train),concept_inds);
maskedBowTrain_learn = maskedBowTrain(:,learn_inds);
maskedBowTrain_concept = maskedBowTrain(:,concept_inds);

faceMasks_train_learn = faceMasks_train(learn_inds);
% sum(train_mil_features)
doHomkermap = false;
train_mil_features = milFeatures(maskedBowTrain_learn,maskedBowTrain_concept,doHomkermap);
test_mil_features = milFeatures(maskedBowTest,maskedBowTrain_concept,doHomkermap);

save milFeatsAll_psix.mat train_mil_features test_mil_features % raw features distances in milFeatsAll.mat

load  milFeatsAll_psix.mat

sig_ = 5;
train_mil_features_1 = double(exp(-train_mil_features/sig_))';
test_mil_features_1 = double(exp(-test_mil_features/sig_))';

means  = mean(train_mil_features_1);
vars = std(train_mil_features_1);

feat_train_all = bsxfun(@minus,train_mil_features_1,means);
feat_train_all = bsxfun(@rdivide,feat_train_all,vars);
feat_test_all = bsxfun(@minus,test_mil_features_1,means);
feat_test_all = bsxfun(@rdivide,feat_test_all,vars);

% feat_test_all(isnan(feat_test_all)) = 0;
% feat_train_all(isnan(feat_train_all)) = 0;
y_train = 2*(t_train(learn_inds)==1)-1;
svm_model = svmtrain(y_train,(feat_train_all),'-t 2 -c 1');

[~,~,ss] = svmpredict(zeros(size(feat_test_all,1),1),double(feat_test_all),svm_model);
sss = ss;

[q,iq] = sort(sss,'descend');
mImage(test_faces_2(iq(1:50)));
mImage(jettify(faceMasks_test(iq(1:50))));

[prec,rec,aps] = calc_aps2(sss,t_test);
plot(rec,prec); title(num2str(aps))

w = svm_model.SVs'*svm_model.sv_coef;

[ww,iww] = sort(w,'descend');
train_faces_2_concept = get_full_image(concept_inds);
train_faces_2_learn = get_full_image(learn_inds);
rr_train_concept = rr_train(:,concept_inds);
[iSector,jImage] = ind2sub([nSectors,length(train_faces_2_concept)],iww);

%( do clustering to see if results make any sense...)
% todo - I am not sure that I'm indexing the right concepts.
% do this using a dummy example!!
% generate many images as a toy classification problem.
for k = 1:length(iSector)    
    k
    curSector = rr_train_concept{jImage(k)}{iSector(k)};
    curImage = train_faces_2_concept{jImage(k)};
    curImage = bsxfun(@times,im2double(curImage),im2double(curSector)) + ...
        .3*bsxfun(@times,im2double(curImage),im2double(1-curSector));
%     imagesc(bsxfun(@times,im2double(curImage),im2double(curSector)));
    imagesc(curImage);
    pause;
end
%%

for k = 1:length(sel_)
    k
    kk = sel_(k);    
    I = get_full_image{kk};
for q = 1:length(r(:));
    imagesc(bsxfun(@times,im2double(I),im2double(r{q})));
    pause(.1);
end
end
% sel_ = find(t_train);
% plot(t_train)
for k = 24:length(sel_)
    k
    kk = sel_(k);    
    I = get_full_image{kk};
    curMask = faceMasks_train{kk};
    r = regionprops(curMask,'Area','PixelIdxList');
    if (length(r)==0)
        warning(['no segments found for image ' num2str(kk)]);
        continue;
    end
     % get the largest segment...
    [~,imax] = max([r.Area]);
    
    
    
    [x,y] = find(curMask);
    xx = mean(x);
    yy = mean(y);
    figure,imagesc(curMask);
    hold on;
    plot(xx,yy,'g+');
    
    
    
    
    Z = zeros(size(curMask));
    Z(r(imax).PixelIdxList) = 1;
   Z = imfilter(im2double(Z),fspecial('gauss',99,3));
    Z = log(Z);
    minVal = -15;
    maxVal = log(.8);
    newMask = (Z<maxVal & Z>minVal);
    clf;
    subplot(1,3,1);imagesc(I);axis image;
    maskedIm = bsxfun(@times,im2double(newMask),im2double(I));
    subplot(1,3,2);imagesc(maskedIm);axis image;
                
    Z = Z>log(.7);
    r = bwboundaries(Z,'noholes');
    
    lengths = cellfun(@length,r);
    [~,ir] = max(lengths);
    r = r(ir);
    subplot(1,3,3);
    imagesc(Z); hold on;
    r = r{1};
    plot(r(:,2),r(:,1),'g','LineWidth',2); axis image;
    pause;
        
end
%%



