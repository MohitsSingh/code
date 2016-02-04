
%%
% subImages
% ids_
% m = readDrinkingAnnotationFile('train_data_to_read.csv');
% newImageData = augmentGT(newImageData,m);
L = load('~/storage/s40_drink_phrase/all.mat');
%%
% allRes = {};
is_valid = true(size(newImageData));
dpm_scores = zeros(length(newImageData),10);
labels = [newImageData.label];
faceScores = [newImageData.faceScore];
faceBoxes = zeros(length(newImageData),4);
phraseRes = cat(1,L.detections{:});
bottleRes = phraseRes(:,2);
phraseRes = phraseRes(:,1);
phraseScores = -inf(size(faceScores));
lm = [newImageData.faceLandmarks];
poses = [lm.c];
dpm_scores2 = zeros(size(dpm_scores));
dpm_models = zeros(size(dpm_scores2));
resDir = '~/storage/dpm_subclass_s40_2';
res = cat(1,detections{:});
%%
for k = 1:length(newImageData)
    curImageData = newImageData(k);
    %     if (curImageData.faceScore < -.6)
    %         is_valid(k) = false;
    %         continue;
    %     end
    k
    %     if (newImageData(k).isTrain && newImageData(k).label)
    %         I = getImage(conf,newImageData(k));
    %         H = computeHeatMap(I,bottleRes(k).boxes,'max');
    % %         figure,imagesc(H)
    % %         figure,imagesc(I)
    %         showboxes(I,bottleRes(k).boxes(1:50,:))
    %     end
    if (~isempty(phraseRes(k).boxes))
        phraseScores(k) = phraseRes(k).boxes(1,end);
    else
        phraseScores(k) = -1;
    end
    for q = 1:size(res,2)
        if (~isempty(res(k,q).boxes))
            dpm_scores(k,q) = res(k,q).boxes(1,end);
        else
            dpm_scores(k,q) = -1;
        end
    end
    %     dpm_scores2(k) = lm(k).dpmRect(end);
    %     dpm_models(k) = lm(k).dpmModel;
    faceBoxes(k,:) = curImageData.faceBox;
end
sz = faceBoxes(:,4)-faceBoxes(:,2);

%
% for k = 1:length(newImageData)
%     k
%     curImageData = newImageData(k);
%     if (curImageData.faceScore < -.6)
%         %         is_valid(k) = false;
%         continue;
%     end
%     if (~newImageData(k).label)
%         M = getSubImage(conf,newImageData(k),f);
%         newImageData(k).sub_image = M;
%     end
% end

%%
close all
is_valid = faceScores >=-.6;
all_feats = [faceScores(:) poses(:) dpm_scores phraseScores'];
isTrain = [newImageData.isTrain]; nTrain = nnz(isTrain); nTest = nnz(~isTrain);
train_sel = isTrain & is_valid;
test_sel = ~isTrain & is_valid;
pos_feats = all_feats(train_sel & labels,:)';
neg_feats = all_feats(train_sel & ~labels,:)';
[x,y] = featsToLabels(pos_feats,neg_feats);
subImages = {newImageData.sub_image};
subImages = subImages(test_sel);
f_test_sel = find(test_sel);
%%
%
%
% classes = {'bottle','cup','straw'};
for iConfig = 1:length(subset_configs)
    finalScores =-inf*ones(length(newImageData),1);
    finalScores(test_sel) = dpm_scores(test_sel,iConfig);
    
    pp = occlusionScores(test_sel)';
    pp(isinf(pp)) = min(pp(~isinf(pp)));
    %     finalScores(test_sel) = finalScores(test_sel)+.1*pp;%+occlusionScores(test_sel)';
    finalScores(test_sel) = finalScores(test_sel);
    %     finalScores(test_sel) = 1*pp;
    %     finalScores = finalScores +.1*double(abs(poses'-7) >=3);
    %     finalScores = finalScores +.01*double(abs(poses'-7));
    %finalScores(test_sel) = f+15*h_test;
    m = showSorted(subImages,finalScores(test_sel),50);
    f_sel = find(test_sel);
    sel_ = ~[newImageData.isTrain];
    labels_ = [newImageData.label];
    labels_(~sel_) = false; % ignore training images.
    %     getResponseMap(conf,newImageData(5275),models{iModel});
    objClasses = subset_configs(iConfig).obj_classes;
    for k = 1:length(labels_)
        if (labels_(k))
            %             labels_(k) = labels_(k) & any(find(cellfun(@any,strfind(objClasses,newImageData(k).extra.objType))));
            labels_(k) = labels_(k) & abs(abs(newImageData(k).extra.obj_orientation) - subset_configs(iConfig).obj_angle) <= 40;
            %             if ~strcmp(newImageData(k).extra.objType,classes{iConfig})
            %                 %                 if (abs(newImageData(k).extra.obj_orientation) < 70)
            %                 labels_(k) = false;
            %                 %                 end
            %             end
        end
    end
    
    all_scores =  finalScores(sel_);
    labels_sel = labels_(sel_);
    % ims, scores, labels, sel_
    [true_imgs,mm] = showResultOrder(allSubIms(sel_),all_scores,[newImageData(~isTrain).label],...
        test_sel(~isTrain));
    
    % find where each labels appears in the final scoring.
    %     [r,ir] = sort(all_scores,'descend');
    %     labels_sel = labels_sel(ir);
    %     true_scores = finalScores(labels_); % scores on real positives.
    %     true_imgs = allSubIms(labels_); % real positive images.
    %     labels_f = find(labels_);
    %     true_imgs_is_inf = paintRule(true_imgs,~isinf(true_scores));
    %     [a,b] = sort(true_scores,'descend');
    %     true_imgs_is_inf = true_imgs_is_inf(b);
    
    %     figure,imshow(newImageData(labels_f(b(13))).sub_image)
    
    %     true_imgs_sorted = showSorted(true_imgs_is_inf,true_scores);
    %     mm = multiImage(true_imgs_is_inf,find(labels_sel));
    % finalScores = finalScores(sel_);
    m_ = 2;n_=2;
    clf; vl_tightsubplot(m_,n_,4,'box','outer');
    
    
    %     getResponseMap
    
    vl_pr(2*labels_(test_sel)-1,finalScores(test_sel));
    vl_tightsubplot(m_,n_,1);%'box','outer'); %subplot(2,2,1);
    imagesc(m); axis image;
    vl_tightsubplot(m_,n_,3);%'box','outer');
    imagesc(mm); axis image;
    nnz(labels_(test_sel))
    
    % number of test images (numer pruned)
    % number of test objects
    
    fprintf(1,'num images with face above threshold / total images: %d/%d\n',...
        nnz(test_sel) , nTest);
    fprintf('number of objects of subclass "%s": %d\n', ...
        subset_configs(iConfig).name,nnz(labels_(test_sel)));
    pause
end
%%
for k = 1:length(newImageData)
    k
    if (test_sel(k))
        %             if (isinf(occlusionScores(k))),continue;end
        curImageData = newImageData(k);
        curImageData.lipScore = lipScores(k);
        f = j2m(occPath, curImageData);
        if (isempty(loaded{k}))
            L = load(f);
            loaded{k} = L;
        else
            L = loaded{k};
        end
        
        if (~isempty(L.rprops))
            seg_scores = score_occluders(curImageData,L);
            occlusionScores(k) = max(seg_scores);
            %            break;
        end
        %         end
    end
end
%%
load straw_scores
%%

finalScores =-inf*ones(length(newImageData),1);
% finalScores(test_sel) = sum(dpm_scores(test_sel,1:2),2)+(phraseScores(test_sel)'>-.5);
finalScores(test_sel) = sum(dpm_scores(test_sel,[1 2 4 8]),2)
% finalScores(test_sel) = max(dpm_scores(test_sel,[1 4]),[],2);
pp = occlusionScores(test_sel)';
pp(isinf(pp)) = min(pp(~isinf(pp)));
% pp(pp < 1) = -1;
face_scores = [newImageData.faceScore];
face_scores = col(face_scores(test_sel));
straw_scores(~test_sel) = -inf;
myPart = .1*(pp);
finalScores(test_sel) = finalScores(test_sel)+0*double(face_scores < .4)+2*(phraseScores(test_sel)');
finalScores(test_sel) = finalScores(test_sel)+myPart+0*straw_scores(test_sel)';
s_dpm = sum(dpm_scores(:,[1]),2);
save ~/storage/misc/occ_and_more.mat face_scores s_dpm occlusionScores phraseScores;

[q,iq] = sort(finalScores,'descend');
% finalScores(test_sel) = f;%sum(dpm_scores(test_sel,[1 2 3]),2);
% show the sorted images, as well as their location
% [true_imgs_f,mm_m,mm_f_only,all_imgs] = showResultOrder(allSubIms(~isTrain),finalScores(~isTrain),[newImageData(~isTrain).label],...
%         test_sel(~isTrain),100);

%     export_fig('/home/amirro/notes/images/2014_04_01/roc.pdf');

%     imshow(mm_m);imshow(mm_f_only);

% mImage(mm_m);mImage(mm_f_only);mImage(all_imgs);export_fig('/home/amirro/notes/images/2014_04_01/top_width_false.pdf');

% m = showSorted(subImages,finalScores(test_sel),50);
sel_ = ~[newImageData.isTrain];
labels_ = [newImageData.label]; %labels_ = labels_(sel_);
% finalScores = finalScores(sel_);
vl_pr(2*labels_(test_sel)-1,finalScores(test_sel));
% vl_pr(2*labels_(~isTrain)-1,finalScores(~isTrain));
%%
% make a histogram of face scores for drinking vs non-drinking.
% train_scores = [newImageData(train_sel).faceScore];
% [n_general,x] = hist(train_scores,40);
% [n_pos] = hist([newImageData(train_sel & labels).faceScore],x);
% n_general = n_general/sum(n_general);n_pos = n_pos/sum(n_pos);
% figure; hold on;stem(x,n_general,'b');stem(x,n_pos,'g');

%%
% add the fisher vectors...
% allFisherFeatures = extractFeatures_(fisherFeatureExtractor,allSubIms);
% save -v7.3 ~/storage/data/cache/allFisherFeats.mat allFisherFeatures
% % % % % load -v7.3 ~/storage/data/cache/allFisherFeats.mat
%%
% % % % % curIms = pos_ims;
% % % % % curIms = cellfun2(@(x)  imResample(x,[80 80],'bilinear'), curIms);
% % % % % [learnParams,conf] = getDefaultLearningParams(conf,1024);
% % % % % fisherFeatureExtractor = learnParams.featureExtractors{1};
% % % % % features_pos = extractFeatures_(fisherFeatureExtractor,curIms);
% % % % % features_neg = allFisherFeatures(:,train_sel & ~labels);
% % % % %
% % % % % classifier = train_classifier_pegasos(features_pos,features_neg,1);
% % % % % feats_test_ = allFisherFeatures(:,test_sel);
% % % % % [~,h_test] = classifier.test(feats_test_);
% % % % %
% % % % % [r,ir] = sort(h_test,'descend');
% % % % % AA = allSubIms(test_sel);
% % % % % feats_test_flip = extractFeatures_(fisherFeatureExtractor,allSubIms(ir(1:100)));
% % % % % h_test_flip = h_test; [~,v] = classifier.test(feats_test_flip);
% % % % % h_test_flip(ir(1:100)) = v;
% % % % % [h_test] = max(h_test,h_test_flip);
%%
classifier_b = Piotr_boosting(double(x),y);
test_feats = all_feats(test_sel,:)';
subImages = {newImageData.sub_image};
subImages = subImages(test_sel);
%%
[Yhat f] = classifier_b.test(test_feats);
%%

[1:length(true_scores);true_scores']
figure,imagesc(multiImage(true_imgs,true));axis image;
% apply dpm to sample imgs... % HERE 5/3/2014
ff = find(labels_);
for ik = 1:length(ff)
    k = ff(ik);
    true_scores(ik)
    respMap = getResponseMap(conf,newImageData(k),models{3});
end


%%
false_images_path = fullfile(conf.cachedir,'false_for_disc_patches.mat');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
%         load ~/storage/misc/imageData_new;
naturalSets = {};
false_images = {};
imageSet = imageData.train;
minFaceScore = -.6;
conf.get_full_image = true;
for k = 1:length(imageSet.imageIDs)
    k
    if (~validImage(imageSet,k,false,minFaceScore))
        continue;
    end
    if (imageSet.labels(k)),continue;end
    currentID = imageSet.imageIDs{k};
    m = getSubImage(conf,newImageData,currentID);
    m = imresize(m,[240 NaN],'bilinear');
    bb = [1 1 size(m,2) size(m,2)];
    dd = .7;
    bb = round(clip_to_image(inflatebbox(bb,[dd dd],'both',false),m));
    false_images{end+1} = cropper(m,bb);
end
save(false_images_path,'false_images');
% end
%%

pos_images = {};
imageSet = imageData.train;
minFaceScore = -.6;
conf.get_full_image = true;
for k = 1:length(imageSet.imageIDs)
    k
    if (~validImage(imageSet,k,true,minFaceScore))
        continue;
    end
    currentID = imageSet.imageIDs{k};
    m = getSubImage(conf,newImageData,currentID);
    m = imresize(m,[240 NaN],'bilinear');
    bb = [1 1 size(m,2) size(m,2)];
    dd = .7;
    bb = round(clip_to_image(inflatebbox(bb,[dd dd],'both',false),m));
    pos_images{end+1} = cropper(m,bb);
    pos_images{end+1} = flip_image(pos_images{end});
end


initialSamples = imageSetFeatures2(conf,pos_images,true,[80 80]);
nSamples = size(initialSamples,2);
[IC,C] = kmeans2(initialSamples',10,...
    struct('nTrial',100,'outFrac',.1,...
    'display',1,'minCl',3));
outliers = IC == -1;
fprintf('fraction of outliers: %0.3f\n',nnz(outliers)/length(IC));
%     IC(outliers) = [];
%     initialSamples(:,outliers) = [];
%     all_ims(outliers) =[];
maxPerCluster = .7;
[curClusters,ims]= makeClusterImages(pos_images',C',IC',initialSamples,[],maxPerCluster);
false_images2 = cellfun2(@(x) imresize(x,[100 NaN]), false_images);
conf.features.winsize = [10 10];
conf.detection.params.init_params.sbin = 8;
clusters = train_circulant(conf,curClusters,false_images2);
w = conf.features.winsize;
figure,imagesc(hogDraw(reshape(clusters(2).w,w(1),w(2),[]),15,1));


conf.get_full_image = true;
conf.max_image_size = inf;
pos_images_test = {};
imageSet = imageData.test;
minFaceScore = -.6;
conf.get_full_image = true;
for k = 1:length(imageSet.imageIDs)
    k
    if (~validImage(imageSet,k,false,minFaceScore))
        continue;
    end
    currentID = imageSet.imageIDs{k};
    m = getSubImage(conf,newImageData,currentID);
    m = imresize(m,[240 NaN],'bilinear');
    bb = [1 1 size(m,2) size(m,2)];
    dd = .7;
    bb = round(clip_to_image(inflatebbox(bb,[dd dd],'both',false),m));
    pos_images_test{end+1} = cropper(m,bb);
    pos_images_test{end+1} = flip_image(pos_images_test{end});
end

[responses]=detect_in_roi(conf,clusters,newImageData,newImageData(r(3)).imageID,true);
% [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
% conf.get_full_image = false;
% conf.max_image_size = 200;

images2 = pos_images_test(1:2:end);
images2 = cellfun2(@(x) imresize(x,[120 NaN],'bilinear'),images2);
%images2 = train_ids(train_labels);
[qq,q] = applyToSet(conf,clusters,images2,[],'','override',true,'visualizeClusters',true);
% for k = 1:length(q)
%     clf; imagesc(q(k).vis);axis image; pause
% end
[A,AA] = visualizeClusters(conf,images2,qq,'add_border',...
    false,'nDetsPerCluster',...
    100,'disp_model',true,'height',64);
displayImageSeries({A.vis});

mImage(AA);

%         [curClusters]= makeClusterImages([],C',IC',initialSamples,[],maxPerCluster);

%%
dd = '/home/amirro/storage/data/drinking_extended/cup/'


[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% [images,inds] = multiRead(conf,dd,'.jpg',[],[],50);
images = train_ids(train_labels);
for k = 1:length(images)
    k
    images{k} = getSubImage(conf,newImageData,images{k});
end
conf.max_image_size = 256;
conf.detection.params.max_models_before_block_method = 10;

images2 = cellfun2(@(x) imresize(x,[240 NaN],'bilinear'),images);

[qq,q] = applyToSet(conf,clusters,images2,[],'','override',true,'visualizeClusters',true);

for k = 1:length(qq)
    clf; imagesc(qq(k).vis);axis image; pause
end


[A,AA] = visualizeClusters(conf,images2,qq,'add_border',...
    false,'nDetsPerCluster',...
    10,'disp_model',true,'height',64);

displayImageSeries({A.vis});

% for k = 1:length(q)
%     clf; imagesc(q(k).vis);axis image; pause
% end

%% ok, this seems to work nicely...

[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
f = find(test_labels);
I = getImage(conf,'drinking_014.jpg');
I = getImage(conf,'looking_through_a_microscope_165.jpg');
conf.get_full_image = true;
imshow(I)
% beer can \/\/\/

for k = 1:length(classifiers)
    %weights_path = '/home/amirro/storage/data/detector_cache/weights/circulant_CUSTOM_%s_weights.mat';
    %         weights_path = '/home/amirro/storage/data/detector_cache/weights/circulant_CUSTOM_%s_weights_refined.mat';
    
    %L = load(sprintf(weights_path,classifiers(k).class));
    
    threshold = 0;
    dTheta = 10;
    
    curClassifier = classifiers(5);
    detection.max_overlap = .6;
    
    %     I1 = I;
    %     for z = 1:3
    %         I1(1:250,:,z) = I(end-249:end,:,z);
    %     end
    %
    %     I2 = imrotate(I,0,'bilinear');
    %     I2 = flip_image(I2);
    I2 = imcrop(I);
    I3 = imresize(I2,[dsize(I2,1:2)].*[1 1],'bilinear');
    I3 = min(1,max(0,I3));
    detection.max_scale = 1;
    detection.min_scale = .01;
    
    %         profile on;
    dTheta = 90;
    %         I1 = imcrop(I);
    I = im2double(imread('/home/amirro/storage/datasets/image_net/images/n02823428/n02823428_2012.JPEG'));
    I = im2double(imread('/home/amirro/storage/datasets/image_net/images/n02823428/n02823428_2522.JPEG'));
    I = rgb2gray(I);
    I = double(I<.5);
    
    I1 = imresize(I,[dsize(I,1:2)].*[1 1]*1,'bilinear');
    %         profile off
    %     I1 = imcrop(I)
    %     I1 = imrotate(I,70,'bilinear','crop');
    dTheta = 10;
    thetaRange = 0:10:350;
    %threshold = 0;
    thetaRange = [-10 10];
    %         all_bbs = detect_rotated(I1,curClassifier,cell_size,features,detection,threshold,dTheta,true);
    dets = detect_rotated2(I1,classifiers,cell_size,features,detection,threshold,thetaRange);
    %        profile viewer
    %         profile viewer
    
    weights = curClassifier.weights; bias = curClassifier.bias;
    w_norm = max(abs(weights(:)));
    weights_ = cat(3,weights,zeros(dsize(weights,1:2)));
    V = hogDraw(weights_,20,1);
    %clf; imagesc(vl_hog('render', 0.4 * single([max(0, weights / w_norm), max(0, -weights / w_norm)])));
    clf; imagesc(hogDraw(single([max(0, weights_ / w_norm), max(0, -weights_ / w_norm)]),15,1));
    axis image;
    pause(.1)
end

initpath;
config;

% record for each image:
% the location of the head.
% the maximal score of a bottle,cup,etc. detector in the vicinity of the
% head.

% create for each detector a "top" point signifying the expected point of
% interface with the mouth.
% This is simply

for k = 1:length(classifiers)
    sz = classifiers(k).object_sz;
    classifiers(k).touch_point = [1 sz(2)/2];
end

% show detections for some objects.
%%
% initpath;
% config;
%%
% load ~/storage/misc/imageData_new;
%
% newImageData = struct('imageID',{},'faceLandmarks',{},'label',{},'faceScore',{},'faceBox',{},'lipBox',{});
%
% imagedatas = {imageData.train,imageData.test};
% t = 0;
% for k = 1:length(imagedatas)
%     imageSet = imagedatas{k};
%     for  kk = 1:length(imageSet.imageIDs)
%         t = t+1;
%         newImageData(t).imageID = imageSet.imageIDs{kk};
%         newImageData(t).faceLandmarks = imageSet.faceLandmarks(kk);
%         newImageData(t).label = imageSet.labels(kk);
%         newImageData(t).faceScore = imageSet.faceScores(kk);
%         newImageData(t).faceBox = imageSet.faceBoxes(kk,:);
%         newImageData(t).lipBox = imageSet.lipBoxes(kk,:);
%     end
% end
%
% save ~/storage/misc/imageData_new imageData newImageData
%
% %%


[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
resDir = '/net/mraid11/export/data/amirro/dpm_s40_sun';
conf.get_full_image = true;
for k = 854:length(train_ids)
    currentID = train_ids{k};
    resPath = fullfile(resDir,strrep(currentID,'.jpg','.mat'));
    if (~exist(resPath,'file')), continue; end
    I = getImage(conf,currentID);
    load(resPath);
    
    for iModel = 1:length(modelResults)
        curClass = modelResults(iModel).class;
        ds = modelResults(iModel).ds;
        u = unique(ds(:,7));
        polys = {};
        scores = [];
        for iu = 1:length(u)
            curDS = ds(ds(:,end)==u(iu),:);
            [w h] = BoxSize(curDS);
            
            sel_ = (w.*h) < .2*(prod(size2(I)));
            if (none(sel_)), continue, end;
            
            curPolys = rotate_bbs(curDS(sel_,:),I,u(iu));
            
            polys = [polys;curPolys(:)];
            scores = [scores;curDS(sel_,6)];
            %             clf; imagesc(I); axis image;hold on;
            %             plotPolygons(curPolys,'g');
            %             pause;
        end
        
        [s,is] = sort(scores,'descend');
        is = is(1:min(100,length(is)));
        polys = polys(is);
        scores = scores(is);
        
        H = computeHeatMap_poly(I,polys,scores,'max');
        M = sc(cat(3,H,im2double(I)),'prob');
        clf; imagesc(M); axis image; title([curClass ', ' num2str(s(1))]);
        pause;
    end
    
end

%%
currentID = curImageData.imageID;
im = getImage(conf,curImageData.imageID);
[regions,~,G] = getRegions(conf,currentID,false);
n = 2;
G = max(G,G');
% [groups,newRegions] = expandRegions(regions,2,[],G);
[r,ro] = chooseRegion(im,regions,.5);
displayRegions(im,r,ro);



%%
model = [];
winsize = conf.features.winsize;

model.w = classifiers(1).weights;
model.w = cat(3,model.w,zeros(dsize(model.w,1:2)));
model.b =  classifiers(1).bias;
model.hg_size = size(model.w);
model.init_params = conf.detection.params.init_params;
models{1}.models_name = 'whatever'; %#ok<*AGROW>
models{1}.model = model;
params = esvm_get_default_params;
params.detect_pyramid_padding = 0;
[resstruct] = esvm_detect(I, models, params);


%%

%%
p = [115 86 ];
bbox = [p p];
bb = round(inflatebbox(bbox,[65 65],'both',true));
I1 = cropper(I,bb);
imshow(I1)

imwrite(I1,'~/nati.png');

[ucm,gpb_thin] = loadUCM(conf,currentID);
figure,imagesc(ucm);
E = edge(im2double(rgb2gray(imresize(I,.5,'bicubic'))),'canny');
imwrite(imresize(I1,1,'bicubic'),'/home/amirro/code/3rdparty/elsd_1.0/2.pgm');
I1 = double(ucm>.1);
imwrite(gpb_thin.^.5,'/home/amirro/code/3rdparty/elsd_1.0/4.pgm');
imwrite(ucm>.1,'/home/amirro/code/3rdparty/elsd_1.0/3.pgm');
imwrite(gpb_thin>0,'/home/amirro/code/3rdparty/elsd_1.0/4.pgm');

%I2 = imfilter(double(gpb_thin>0),fspecial('gauss',5,.3));
I2 = double(gpb_thin>0);
% imwrite(I2,'/home/amirro/code/3rdparty/elsd_1.0/2.pgm');
imwrite(I2,'/home/amirro/code/3rdparty/elsd_1.0/2.pgm');

figure,imagesc(imresize(I2,5));
figure,imagesc(imresize(I2,.5));
elsd_gpb(conf,{'drinking_168.jpg'});
%%
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% tt = train_ids(train_labels);

conf.get_full_image = false;
% show elsd results...

% some interesting cases - 802,809,821,823

for q = 1:length(train_ids)
    
    if (imageData.train.faceScores(q) < -.6)
        continue;
    end
    
    isIMG  = find((cellfun(@any,strfind(imageIDS,train_ids{q}))));
    objectsInImage = gtParts(isIMG);
    isObj = cellfun(@any,strfind(objectsInImage,'cup'));
    if (none(isObj))
        continue;
    end
    
    if (none(strfind(train_ids{q},'drink')))
        %             continue;
    end
    %
    I = imread(getImagePath(conf,train_ids{q}));
    train_ids{q}
    elsd_file = fullfile(elsd_output_dir,strrep(train_ids{q},'.jpg','.txt'));
    A = dlmread(elsd_file);
    [I_cropped,I_rect] = getImage(conf,train_ids{q});
    %     I_cropped = im2double(I);
    lineSegFile = fullfile(conf.lineSegDir,strrep(train_ids{q},'.jpg','.mat'));
    %     L = load(lineSegFile);
    clf; imshow(1-0*I_cropped); axis image; hold on;
    
    A = dlmread(elsd_file);
    [lines_,ellipses_] = parse_svg(A,I_rect(1:2));
    
    % also get the lines...
    d = findDegenerateEllipses(ellipses_,3);
    
    plot_svg(lines_,ellipses_,d);
    % decide an ellipse is degenerate if it's radius is larger than the
    % image's diagonal
    %         d = d | ellipses_(:,3) > .5*norm(dsize(I_cropped,1:2));
    
    %     [ucm,gpb_thin] = loadUCM(conf,train_ids{q});
    %     [edgelist edgeim] = edgelink(ucm>.1, []);
    %     segList = seglist2segs(edgelist);
    %     segList = segList(:,[2 1 4 3]);
    %     segList=2+bsxfun(@minus,segList,I_rect([1 2 1 2]));
    %     lines_ =[ lines_;segList];
    
    %     test_geometry(I_cropped,lines_,ellipses_,imageData.train,q);
    
    follow_geometry(I_cropped,lines_,ellipses_,imageData.train,q);
    
    q
    pause
    continue;
    %    figure,imagesc(I);axis image;
    % %     hold on; drawedgelist(L.edgelist,[],1,'rand')
    %     seglist = lineseg(L.edgelist,1);
    %     hold on; drawedgelist(seglist,[],1,'rand')
    
    % now define geometric properties of configurations of ellipses / lines
    % which satisfy "a cup"
    
    %     I = imcrop(I_cropped);
    %     imwrite(I_cropped,'/home/amirro/code/3rdparty/elsd_1.0/1.pgm');
    
    %     I = imcrop(I_cropped);
    %     addpath('~/code/3rdparty/face-release1.0-basic/');
    %     detect_landmarks_99(conf,{I},1);
    
    faceRect = imageData.train.faceBoxes(q,:);
    faceRect = makeSquare(faceRect);
    faceRect = inflatebbox(faceRect,[1 1],'both',false);
    %     regions = chooseRegion(I,regions,.5);
    %     [regions,regionOvp,G] = getRegions(conf,train_ids{q},1);
    %     displayRegions(im2double(I),regions);
    %     r = double(regions{1});
    %     r = imfilter(r,fspecial('gauss',5,1.7));
    %     imwrite(imresize(r,.25),'/home/amirro/code/3rdparty/elsd_1.0/1.pgm');
    %
    
    %TODO -
    
    %     figure,imshow(I_cropped);
    %     [ucm,gPb_thin] = loadUCM(conf,train_ids{q});
    %     figure,imshow(gPb_thin);
    %     imagesc(ucm);
    
    %     I1 = cropper(I_cropped,faceRect);
    %     imwrite(I1,'/home/amirro/code/3rdparty/elsd_1.0/1.pgm');
    %
    %     xlim(faceRect([1 3]));
    %     ylim(faceRect([2 4]));
    
    
    %      ext_ = 'png';
    %     saveas(gcf,fullfile(resPath,strrep(tt{q},'jpg',ext_)),ext_);
end

for k = 1:length(train_ids)
    %     if (~train_labels(k)), continue, end;
    imageID = train_ids{k};
    out_dir = '~/s40_elsd_regions';
    %     imageID=  'blowing_bubbles_006.jpg';
    resPath = fullfile(out_dir,strrep(imageID,'.jpg','.mat'));
    if (~exist(resPath,'file')), continue, end;
    L = load(resPath);
    I = getImage(conf,imageID);
    [lines_,ellipses_] = parse_svg(L.A);
    clf,subplot(1,2,1);imagesc(1-I*0);axis image; hold on;
    
    %     figure,imagesc(I);axis image; hold on;
    plot_svg(lines_,ellipses_);
    subplot(1,2,2); imagesc(I); axis image; hold on;
    
    %           [ucm,gPb_thin] = loadUCM(conf,imageID);
    imageID
    pause
end

imwrite(imresize(double(gPb_thin>.1),1),'/home/amirro/code/3rdparty/elsd_1.0/2.pgm');

imshow(cumsum(cumsum(gPb_thin,1),2),[])


imwrite(I,'/home/amirro/code/3rdparty/elsd_1.0/3.pgm');

elsd_with_regions(conf,train_ids(train_labels));
elsd_with_regions(conf,train_ids);
elsd_with_regions(conf,train_ids);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
elsd_with_regions(conf,test_ids);

%%
cmd = 'elsd 1.pgm 1.res'
cd '/home/amirro/code/3rdparty/elsd_1.0/';
[status,result] = system(cmd);

%%
outDir = '~/ucm_demo';
mkdir(outDir);
for k = 1:length(train_ids)
    
    if (train_labels(k))
        [ucm,gpb_thin] = loadUCM(conf,train_ids{k});
        I = sc(gpb_thin,'gray');
        imwrite(I,strrep(fullfile(outDir,train_ids{k}),'.jpg','.png'));
    end
end
%%
outDir = 'ucm_and_elsd';
d = dir(fullfile('~/ucm_demo/*.png'));
for k = 1:length(d)
    k
    a1 = imread(fullfile('~/ucm_demo/',d(k).name));
    a1 = im2double(a1);
    %     a1(a1<.1) = 0;
    a1 = 1-a1;
    a2 = im2double(imread(fullfile('/home/amirro/storage/s40_elsd_output/converted/',d(k).name)));
    I = [a1 zeros(size(a1,1),3,3) a2];
    imwrite(I,fullfile(outDir,d(k).name));
    
end




% initpath;
% config;


%% dsp matching (grauman)
addpath('/home/amirro/code/3rdparty/dsp-code');
facesPath = fullfile('/home/amirro/mircs/experiments/experiment_0008/faces.mat');
imagesc(multiImage(faces.train_faces(1:100),true,false)); axis image;

mImage(L_imgs.ims(nns_train(k,1:5)));
%%

% I = faces.train_faces{1};
k = 830

nnnn = L_imgs.ims(nns_train(k,:));

% im1 = nnnn{1};

im2 = nnnn{1};
im1 = faces.train_faces{k};
im1 = imResample(im1,1,'bilinear');
im2 = imResample(im2,1,'bilinear');
im2 = imresize(im2,[size(im1,1),size(im1,2)],'bilinear');
clf; subplot(1,2,1); imagesc(im1); axis image;
subplot(1,2,2); imagesc(im2); axis image;
% pause

%%
% extract SIFT
[sift1, bbox1] = ExtractSIFT(im1, pca_basis, sift_size);
[sift2, bbox2] = ExtractSIFT(im2, pca_basis, sift_size);
im1 = im1(bbox1(3):bbox1(4), bbox1(1):bbox1(2), :);
im2 = im2(bbox2(3):bbox2(4), bbox2(1):bbox2(2), :);
% Match
tic;
[vx,vy] = DSPMatch(sift1, sift2);



t_match = toc;

warp21=warpImage(im2double(im2),vx,vy); % im2 --> im1

figure,
subplot(2,2,1);
imshow(im1);
title('image1');
subplot(2,2,3);
imshow(im2);
title('image2');
subplot(2,2,2);
imshow(warp21);
title('warp 2-->1');

%%
hold on; quiver(vx,vy,0,'g')
% subplot(2,2,4);
% imshow(seg);
% title('label transfer 2-->1');

%% self-similarity
addpath('/home/amirro/code/3rdparty/ssdesc-cpp-1.1.1');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
facesPath = fullfile('/home/amirro/mircs/experiments/experiment_0008/faces.mat');
%     load '/home/amirro/mircs/experiments/experiment_0001/sals_new.mat';
%     L = load('/home/amirro/mircs/experiments/experiment_0001_improved/exp_result.mat');
load(facesPath);

I = faces.train_faces{802};
imshow(I);
params.patch_size = 5;
params.desc_rad = 40;
params.nrad = 3;
params.nang = 12;
params.var_noise = 300000;
params.saliency_thresh = 0.7;
params.homogeneity_thresh = 0.7;
params.snn_thresh = 0.85;
I = imresize(I,2);;
tic
[resp, draw_coords, salient_coords, homogeneous_coords, snn_coords] = mexCalcSsdescs(double(I),params);
toc

clf; imagesc(I); axis image; hold on;
plot(draw_coords(1,:),draw_coords(2,:),'r.');

clf; imagesc(I); axis image; hold on;
plot(homogeneous_coords(1,:),homogeneous_coords(2,:),'r.');




%% hand detection results.
initpath;
config;
conf.get_full_image = 1;
%%
%L = load('~/storage/hands_s40/top_bbs_test_ita.mat');
% L = load('~/storage/hands_s40/top_bbs_train_ita.mat');
% L = load('~/storage/hands_s40/top_bbs_test.mat');
L = load('~/storage/hands_s40/top_bbs_train.mat');
top_bbs = L.top_bbs_train;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels] = getImageSet(conf,'test');
%%
ids = train_ids;
labels = train_labels;
conf.get_full_image = true
bb = {};
imgs = {};
for k = 1:length(ids)
    k
    %     if (~labels(k))
    %         continue;
    %     end
    if (~any(strfind(ids{k},'wav')))
        continue;
    end
    %     if (~strcmp(ids{k},'drinking_176.jpg'))
    %         continue;
    %     end
    %
    
    resName = fullfile('~/storage/hands_s40_ita',strrep(train_ids{k},'.jpg','.mat'));
    load(resName);
    %     boxes = top_bbs{k};
    if (size(boxes,1)>0)
        bb{end+1} = boxes(1,:);
        I = getImage(conf,ids{k});
        imgs{end+1} = cropper(I,round(boxes(1,:)));
    end
    clf; imagesc2(getImage(conf,ids{k})); axis image; hold on;
    %plotBoxes(boxes(1:min(5,size(boxes,1)),:),'g','LineWidth',2);
    %     plotBoxes(boxes(1,:),'g','LineWidth',2);
    plotBoxes(boxes(1:min(3,size(boxes,1)),:),'g','LineWidth',2);
    boxes(1,end)
    pause;
    %showboxes_(getImage(conf,train_ids{k}),L.boxes);
end
%%
bb = cat(1,bb{:});
showSorted(imgs,bb(:,6),1000);

% [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% %%
% for k = 100:length(train_ids)
%     if (~train_labels(k))
% %         continue;
%     end
%     resName = fullfile('~/storage/hands_s40_ita',strrep(train_ids{k},'.jpg','.mat'));
%     L = load(resName);
%     clf; imagesc(getImage(conf,train_ids{k})); axis image; hold on;
%     plotBoxes2(L.boxes(1:min(3,size(L.boxes,1)),[2 1 4 3]),'g','LineWidth',2);
%     pause;
%     %showboxes_(getImage(conf,train_ids{k}),L.boxes);
%
% end
%%
conf.imgDir = '/home/amirro/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages';

d = dir(fullfile(conf.imgDir,'*.jpg'));
for k = 1:length(d)
    k
    fName = fullfile(conf.imgDir,d(k).name);
    imwrite(imread(fName),fName,'Quality',100);
    %     iInfo = imfinfo(fName);
    %     if (~strcmp(iInfo.Format,'jpg'))
    %         disp (fName);
    %         break;
    %     end
end

%% check how well I can enstimate pose using the dpm detection....
load ~/mircs/experiments/common/faces_cropped_new.mat;
load imageData_new;

imageSet = imageData.train;
faceSet = faces.train_faces;

dpmModels = [imageSet.faceLandmarks.dpmModel];
scores_ = imageSet.faceScores;

comps = [imageSet.faceLandmarks.c];

u = unique(comps);

for iu = 1:length(u)
    uu = u(iu)
    faces = faceSet(comps==uu);
    scores = scores_(comps==uu);
    showSorted(faces,scores,10);
    pause;
    close all
end


%% show some of the face detections on the phrasal recognition dataset
load ~/storage/UIUC_PhrasalRecognitionDataset/VOC3000/groundtruth;
%tr = find(strcmp('person_drinking_bottle', groundtruth(:,2)));

conf.imgDir = '~/storage/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages/';
tr = 1:length(groundtruth);
% k = 1:length(tr);
%%
close all;
for k = 1:length(tr)
    k
    clf;
    if (k>1) && strcmp(groundtruth{tr(k),1},groundtruth{tr(k-1),1})
        continue;
    end
    [I,xmap] = imread(fullfile(conf.imgDir,[groundtruth{tr(k),1} '.jpg' ]));
    if (length(size(I))==2)
        I = repmat(I,[1 1 3]);
    end
    
    % load the face detection results...
    groundtruth{tr(k),1}
    if (~exist(fullfile('~/storage/faces_phrasal',[groundtruth{tr(k),1} '.mat'])))
        continue;
    end
    L = load(fullfile('~/storage/faces_phrasal',[groundtruth{tr(k),1} '.mat']));
    
    
    
    %     clf;imagesc(getImage(conf,image_ids{k}));axis image; hold on;
    % load the landmarks file according to the image name.
    imageName = [groundtruth{tr(k),1} '.jpg'];
    %     imageInd =  find(cellfun(@(x)~isempty(x),strfind(image_ids,imageName)));
    
    % show landmarks...
    
    dss = cat(1,L.res.ds);
    dss = dss(:,1:6);
    top = nms(dss,.5);
    dss = dss(top,:);
    %[s,is] = sort(dss(:,6),'descend');
    %     is = is(1:min(length(is),1));
    
    %     dss = dss(is,:);
    clf;
    imagesc(I); axis image; hold on;
    plotBoxes2(dss(:,[2 1 4 3]),'r');
    if (size(dss,1)>2)
        plotBoxes2(dss(3,[2 1 4 3]),'b','LineWidth',2);
    end
    if (size(dss,1)>1)
        plotBoxes2(dss(2,[2 1 4 3]),'m','LineWidth',2);
    end
    plotBoxes2(dss(1,[2 1 4 3]),'g','LineWidth',2);
    
    %     showboxes(I,dss);
    
    %     clf; imagesc(I); axis image;
    %     hold on; plotBoxes2(dss(:,[2 1 4 3]),'g','LineWidth',2);
    %     plotBoxes2(faceLandmarks(imageInd).xy(:,[2 1 4 3]),'g');
    disp('loaded');
    pause
end


%%
cd /home/amirro/code/3rdparty/voc-release5
for k = 1:10
    load(sprintf('models/face_big_%d_final.mat',k));
    models(k) = model;
end
startup
iModel = 1

dss = {};
for iRot =-60:10:60
    
    iRot
    curDS = detectRotated(I,models(iModel),-1,iRot);
    
    %             pause
    if (~isempty(curDS))
        curDS = [curDS,repmat(iRot,size(curDS,1),1)];
        dss{end+1} = curDS;
    end
end

ds = cat(1,dss{:});

%         showboxes(

res(iModel).ds = ds;
%     end
save(resFileName,'res');
% end

% fprintf('\n\n\nFINISHED\n\n\n!\n');



%%

image_ids = {};
for k = 1:length(d)
    image_ids{k} = d(k).name;
end
landmarkDir = '~/storage/landmarks_phrasal';
[faceLandmarks,allBoxes_complete,faceBoxes_complete] = collectResults(conf,image_ids,landmarkDir);


scores = [faceLandmarks.s];
[s,is] = sort(scores,'descend');
for q = 1:length(image_ids)
    k = is(q);
    clf;imagesc(getImage(conf,image_ids{k}));axis image; hold on;
    plotBoxes2(faceLandmarks(k).xy(:,[2 1 4 3]),'g');
    pause;
end



%% try to make some good of the hands images....

opts.hands_locs_suff = 'hands_locs';
opts.hands_images_suff = 'hands_imgs';
inputDir = '/home/amirro/storage/data/Stanford40/JPEGImages';
actionsFileName = '/home/amirro/storage/data/Stanford40/ImageSplits/actions.txt';
[A,ii] = textread(actionsFileName,'%s %s');
A = A(2:end);

% 3 -> brushing teeth
% 9 -> drinking
% 24 -> phoning
% 40 -> writing on a book
% 31 -> taking a photo
% 32 -> texting message
% k =conf.class_enum.DRINKING;


dirs = dir(fullfile(inputDir,'*hands_locs'));

hand_locs = struct('sourceImage',{},'rects',{});

k = 0;
for iDir = 1:length(dirs)
    iDir
    curDir = fullfile(inputDir,dirs(iDir).name);
    curFiles = dir(fullfile(curDir,'*.mat'));
    for iFile = 1:length(curFiles)
        iFile
        curName = curFiles(iFile).name;
        if (~isempty(strfind(curName,'_rec')))
            continue;
        end
        [~,name,~] = fileparts(curName);
        k = k+1;
        hand_locs(k).sourceImage = name;
        handsFile = fullfile(curDir,curName);
        
        L = load(handsFile);
        
        rs = {};
        
        if (isempty(L.rects));
            delete(handsFile);
            continue;
        end
        
        for q = 1:length(L.rects)
            if (~isempty(L.rects(q).left))
                rs{end+1} = [L.rects(q).left.tl L.rects(q).left.br];
            end
            if (~isempty(L.rects(q).right))
                rs{end+1} = [L.rects(q).right.tl L.rects(q).right.br];
            end
        end
        
        rects =  cat(1,rs{:});
        if (~isempty(rects))
            
            %         rects = hand_locs(k).rects;
            xmin = min(rects(:,[1 3]),[],2);
            xmax = max(rects(:,[1 3]),[],2);
            ymin = min(rects(:,[2 4]),[],2);
            ymax = max(rects(:,[2 4]),[],2);
            
            rects = [xmin ymin xmax ymax];
            
            hand_locs(k).rects = rects;
        end
    end
end


% save hand_locs hand_locs
%
% for q = 1:10:length(hand_locs)
%     k
%     I = getImage(conf,[hand_locs(q).sourceImage '.jpg']);
%     clf; imagesc(I); axis image; hold on;
%     if (~isempty(hand_locs(q).rects))
%         plotBoxes2(hand_locs(q).rects(:,[2 1 4 3]),'g','LineWidth',2);
%     end
%     pause;
% end


%% get some drinking hands images....
conf.get_full_image = true;
conf.class_subset = conf.class_enum.DRINKING;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
true_ids = train_ids(train_labels);
goods = true(size(hand_locs));
mm ={};
goodRects = {};
for k = 1:length(hand_locs)
    k
    imName = [hand_locs(k).sourceImage '.jpg'];
    r = find(cellfun(@(x) ~isempty(x),strfind(true_ids,imName)));
    if (isempty(r))
        goods(k) = false;
    else
        I = getImage(conf,[hand_locs(k).sourceImage '.jpg']);
        
        if (~isempty(hand_locs(k).rects))
            %             break;
            
            
            rects = hand_locs(k).rects;
            %
            %         xmin = min(rects(:,[1 3]),[],2);
            %         xmax = max(rects(:,[1 3]),[],2);
            %         ymin = min(rects(:,[2 4]),[],2);
            %         ymax = max(rects(:,[2 4]),[],2);
            %
            %         rects = [xmin ymin xmax ymax];
            % % %
            %                     hand_locs(k).rects = rects;
            
            
            
            
            % qq =multiCrop(conf,{I},rects);
            %         mm = [mm,qq];
            
            
            %             clf; imagesc(mm{1}); axis image; p
            % clf; imagesc(I); axis image; hold on;
            %             plotBoxes2(hand_locs(k).rects(:,[2 1 4 3]),'g','LineWidth',2);
            %             pause;
        end
    end
end

%%
mImage(mm);

mm_flipped = cellfun2(@(x) flip_image(x),mm);

mmm = [mm,mm_flipped];

conf.features.vlfeat.cellsize = 8;
X = imageSetFeatures2(conf,mmm,true,[64 64]);
[IDX,C] = kmeans2(X',5,struct('nTrial',1));
maxPerCluster = 300;
[clusters,ims_,inds] = makeClusterImages(mmm,C',IDX',X,'drinking_hands_1',maxPerCluster);


%%

%
%
% for iClass_subset = 1:40
%     currentTheme = A{k};
%     % currentTheme
%     conf.class_subset = iClass_subset
%     hands_locs_dir = fullfile(inputDir,[currentTheme '_' opts.hands_locs_suff]);
%     [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
%     T = train_ids(train_labels);
%     conf.get_full_image = true;
% %%
% for k  = 1:length(train_ids)
%     outFileName = fullfile(hands_locs_dir,strrep(T{k},'.jpg','.mat'));
%
%     L = load(outFileName);
%
%     rs = {};
%     for q = 1:length(L.rects)
%         if (~isempty(L.rects(q).left))
%             rs{end+1} = [L.rects(q).left.tl L.rects(q).left.br];
%         end
%         if (~isempty(L.rects(q).right))
%             rs{end+1} = [L.rects(q).right.tl L.rects(q).right.br];
%         end
%     end
%
%     rs =  cat(1,rs{:});
%     if (isempty(rs))
%         continue;
%     end
% %     r1 = [L.rects.left.tl L.rects.left.br];
% %     r2 = [L.rects.right.tl L.rects.left.br];
%     I = getImage(conf,T{k});
%     clf; imshow(I);
%     hold on; plotBoxes2(rs(:,[2 1 4 3]),'g','LineWidth',2);
%     pause;
% end
%%


%% re-collect the landmark localizations....

% load ~/storage/face_det_new_train.mat; %% bb_train


[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
m = train_ids;
conf.get_full_image = false;
toBreak = false;

for k = 1:length(m);
    currentID = m{k};
    if (isempty(strfind(currentID,'drinking')))
        continue;
    end
    resFile = fullfile('~/storage/landmarks_s40',strrep(currentID,'.jpg','.mat'));
    if (~exist(resFile,'file'))
        continue;
    end
    load(resFile);
    
    curBoxes = bb_train(bb_train(:,11)==k,:);
    
    
    I = getImage(conf,currentID);
    clf;
    subplot(1,2,1); imagesc(I); axis image; hold on;
    bbs_ = cat(1,landmarks.hogRect);
    plotBoxes2(bbs_(:,[2 1 4 3]));
    landmarkScores = -ones(size(landmarks));
    hogScores = zeros(size(landmarks));
    
    for iL = 1:length(landmarks)
        curLandmark = landmarks(iL);
        hogScores(iL) = curLandmark.hogRect(12);
        if (~isempty(curLandmark.bs))
            landmarkScores(iL) = curLandmark.bs.s;
            %             toBreak = true;
            %             break;
        end
    end
    
    [s,is] = sort(landmarkScores,'descend');
    if (s(1)==-1)
        continue;
    end
    
    bestLandmark = landmarks(is(1));
    bestBB = bbs_(is(1),:);
    
    bbSize = bestBB(3:4)-bestBB(1:2);
    
    resizeFactor = bbSize(1)/256;
    xy = bsxfun(@plus,bestLandmark.bs.xy*resizeFactor,bestBB([1 2 1 2]));
    
    xy_c = boxCenters(xy);
    plot(xy_c(:,1),xy_c(:,2),'g.');
    pause;
    %
    %     if (toBreak)
    %         break;
    %     end
end


%%
% correct the landmark detection....
imageData = initImageData;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
train_true = train_ids(train_labels);
for k = 7:length(train_true)
    currentID = train_true{k};
    k
    
    f = find(cell2mat(cellfun2(@(x) ~isempty(x), strfind(imageData.train.imageIDs,currentID))));
    if (isempty(f))
        find(cell2mat(cellfun2(@(x) ~isempty(x), strfind(imageData.test.imageIDs,currentID))));
    end
    if (isempty(f))
        conf.get_full_image = false;
        I = getImage(conf,currentID);
        %         I = imcrop(I);
        clf; imagesc(I); axis image; pause;
        %          imshow(imrotate(I,30))
        detect_landmarks(conf,{imrotate(I,0,'bilinear')},2,false);
        %           detect_landmarks(conf,{imrotate(I,0,'bilinear')},1,true);
        detect_landmarks(conf,{imrotate(I,50,'bilinear')},1,false);%
    end
end



%%
load newFaceData.mat;

train_faces = cat(4,train_faces{:});
M = mat2cell2(train_faces,[1 1 1 size(train_faces,4)]);
[X,sz] = imageSetFeatures2(conf,M,true,[64 64]);
[IDX,C] = kmeans2(X',10,struct('nTrial',1));
%       clusters = makeClusters(X,[]);
[clusters,ims] = makeClusterImages(M,C',IDX',X,'dir1');

for k = 1:10
    k
    clf;montage2(train_faces(:,:,:,IDX==k),struct('hasChn',1));
    pause;
end

%%
% imageData = initImageData(conf);

imageSet = imageData.train;
conf.get_full_image = true;
cur_t = imageSet.labels;
for k = 1:length(cur_t)
    % find the contour of the face! :-)
    
    if (~cur_t(k))
        %         continue;
    end
    currentID = imageSet.imageIDs{k};
    I = getImage(conf,currentID);
    xy = imageSet.faceLandmarks(k).xy;
    xy_c = boxCenters(xy);
    clf;
    imagesc(I);
    hold on;
    plot(xy_c(:,1),xy_c(:,2),'r.');
    
    k = convhull(xy_c);
    plot(xy_c(k,1),xy_c(k,2),'g--');
    pause;
end

%%


% learn a new face detector - again.
[images,inds] = multiRead(conf,'/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped','jpg',[],[64 64],inf);
conf.features.vlfeat.cellsize = 8;
fHandle = @(x) col(single(vl_hog(imResample(im2single(x),[64 64]),conf.features.vlfeat.cellsize,'NumOrientations',9)));
X = fevalImages(fHandle,{},'/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped','face','jpg',1);


d = dir('/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped/*.jpg');
d = d(1:3:end);

ims = {};%zeros([64 64 3 length(d)]);
for k = 1:length(d)
    k
    ims{k} = imResample(imread(fullfile('/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped/',d(k).name)),[80 80],'bilinear');
end


save ~/storage/X.mat X
% for k = 1:length(images)
%     k
%    I = images{k};
%    if (length(size(I)) < 3)
%        I = cat(3,I,I,I);
%    end
%    images{k} = I;
% end
% save ~/storage/many_faces.mat images
% montage2(images(:,:,:,1:10:end),struct('hasChn',true));

% M = mat2cell2(images,[1 1 1 size(images,4)]);
% [X,sz] = imageSetFeatures2(conf,M{1},true,[64 64]);
X_ = X(:,1:3:end,:);
[IDX,C] = kmeans2(X_',5,struct('nTrial',1));

conf.detection.params.init_params.sbin = 8;
conf.features.winsize = [8 8];
conf.detection.params.detect_add_flip = 0;
conf.detection.params.detect_min_scale = .5;
conf.detection.params.detect_levels_per_octave =8;

for k = 1:5
    k
    imagesc(showHOG(conf,C(k,:).^2)); axis image; pause;
end

%       clusters = makeClusters(X,[]);

maxPerCluster = 100;
% [II,I1,I2,I3] = montage2(cat(4,ims{1:5}),struct('hasChn',true));
[clusters] = makeClusterImages(ims,C',IDX',X_,'dir1',maxPerCluster);

for k = 1:5
    k
    imagesc(showHOG(conf,clusters(k).w.^2));axis image; pause;
end

% clusters_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','faces_new','override',true,'C',.001);
clusters_trained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','faces_new','override',false,'C',.001);

for k = 1:5
    k
    imagesc(showHOG(conf,clusters_trained(k)));axis image; pause;
end



%% refine using the detections....

d = dir('/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped/*.jpg');
d = d(1:7:end);
conf.detection.params.detect_add_flip = 0;
conf.detection.params.detect_min_scale = .8;
conf.detection.params.detect_levels_per_octave =8;

ims = {};%zeros([64 64 3 length(d)]);
for k = 1:length(d)
    k
    ims{k} = imResample(imread(fullfile('/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped/',d(k).name)),[80 80],'bilinear');
    if (length(size(ims{k}))==2)
        ims{k} = repmat(ims{k},[1 1 3]);
    end
end

% mImage(ims(1:10:end));

qq_train_refine = applyToSet(conf,clusters_trained,ims,[],'face_train_refine','override',false,'disp_model',true,...
    'uniqueImages',true,'nDetsPerCluster',1,'visualizeClusters',false,'toSave',false);


% extract the new sub-images...

newClusters = clusters_trained;

for k = 1:length(qq_train_refine)
    k
    mmm = multiCrop(conf,ims,qq_train_refine(k).cluster_locs(1:200,:),[64 64]);
    newClusters(k) = makeCluster(imageSetFeatures2(conf,mmm,true,[64 64]),[]);
    mImage(mmm); pause;
    imshow(showHOG(conf,newClusters(k)));
    pause
    close all;
end

prepareForDPM



clusters_trained_refined = train_patch_classifier(conf,newClusters,getNonPersonIds(VOCopts),'suffix','faces_new_refined','override',true,'C',.001);


for k = 1:5
    k
    imagesc(showHOG(conf,clusters_trained_refined(k).w));axis image; pause;
end


conf.get_full_image = false;
conf.detection.params.detect_add_flip = 0;
conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_levels_per_octave =8;

qq_train = applyToSet(conf,clusters_trained_refined,train_ids,[],'face_train','override',true,'disp_model',true,...
    'uniqueImages',true,'nDetsPerCluster',10,'visualizeClusters',false);
bb_train = cat(1,qq.cluster_locs);
m = train_ids(train_labels);

save ~/storage/face_det_new_train.mat bb_train


bb_test = cat(1,qq.cluster_locs);

save ~/storage/face_det_new_test.mat bb_test


% now run the parallel landmark localization...

% m = train_ids(randperm(length(train_ids)));
bb = cat(1,qq.cluster_locs);
% bb = bb(:,[1:4 12]);
%%
for k = 1:length(m)
    I = getImage(conf,m{k});
    f = find(bb(:,11)==k);
    clf; imagesc(I); hold on;
    plotBoxes2(bb(f,[2 1 4 3]),'g','LineWidth',2);
    pause;
    
end
%%

conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_max_scale = 1;
conf.detection.params.detect_levels_per_octave =8;
m = train_ids(train_labels);
for k = 28:length(m)
    % for k = 9
    k
    I = getImage(conf,m{k});
    I = I(1:floor(.6*end),:,:);
    I = I(:,floor(.1*end):ceil(.9*end),:);
    %     I = I(4:end,4:end,:);
    sz = size(I);
    [sz_,isz] = min(sz(1:2));
    
    %     if (sz_ < 100)
    %         I = imresize(I,100/sz_,'bilinear');
    %     end
    
    %     I = imcrop(I);
    
    %     I = imrotate(I,45,'bilinear','loose');
    
    I = min(1,max(0,I));
    
    qq = applyToSet(conf,clusters_trained,{I},[],'faces_new_check1','override',true,'disp_model',true,...
        'uniqueImages',true,'nDetsPerCluster',inf,'rotations',[0],'visualizeClusters',false);
    bb = cat(1,qq.cluster_locs);
    
    
    %     bb = [35 90 45 100];
    %     bb(13) = 40;
    
    bb = fix_bb_rotation(bb,I);
    
    
    isFlipped = bb(:,7);
    %     bb(isFlipped,:) = flip_box(bb(isFlipped,:),size(I));
    %     f = find(bb(:,11)==k);
    %     I = getImage(conf,m{k});
    clf; subplot(1,2,1); imagesc(I); axis image; hold on;
    plotBoxes2(bb(:,[2 1 4 3]),'g','LineWidth',2);
    bb = bb(:,[1:4 12]);
    %     bb(:,5) = normalise(bb(:,5));
    bb(:,1:4) = round(clip_to_image(bb(:,1:4),I));
    
    
    HH = computeHeatMap(I,bb,'max');
    subplot(1,2,2); imagesc(HH); axis image;%colorbar
    pause;
end
%%
% mm = m(qq(1).cluster_locs(14,11));
qq = applyToSet(conf,clusters_trained,mm,[],'faces_new_check1','override',false,'disp_model',true,...
    'uniqueImages',true,'nDetsPerCluster',inf);
I = getImage(conf,mm{1});
I = I(1:end/2,:,:);

qq = applyToSet(conf,clusters_trained,{I},[],'faces_new_check1','override',true,'disp_model',true,...
    'uniqueImages',true,'nDetsPerCluster',inf);

bb = cat(1,qq.cluster_locs);
bb = bb(:,[1:4 12]);

bb = clip_to_image(bb,I);
HH = computeHeatMap(I,bb,'max');
imshow('faces_new_check1.jpg');
figure,imshow(HH,[]);
imagesc(I); hold on; plotBoxes2(bb(:,[2 1 4 3]));
%%figure,imshow(getImage(conf,m{qq(1).cluster_locs(14,11)}))





%%
for k = 1:10
    k
    imagesc(showHOG(conf,clusters_trained(k))); axis image; pause;
end
% images = cat(4,images{:});
%%

%% I want to learn the appearance of different cups - use jittering to do so.

% cupImages = getGtImages(conf,gt_cup_aligned);
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[groundTruth,partNames] = getGroundTruth(conf,train_ids,train_labels);
conf.get_full_image = true;
gtParts = {groundTruth.name};
isObj = cellfun(@any,strfind(gtParts,'cup'));
cupImages = getGtImages(conf,groundTruth(isObj),false,1);

for k = 1:length(cupImages)
    curImage = cupImages{k};
    cupImages{k} = imResample(curImage,[64 64],'bilinear');
end

alignByJittering(conf,cupImages);


% make a set of "exemplar svms" from these.
hog_gauss_data = hog_gauss();
%%
X_w = {};

for k = 1:length(cupImages)
    curImage = cupImages{k};
    curImage = imResample(curImage,[64 64],'bilinear');
    x = vl_hog(im2single(curImage),8);
    
    x_1 = x(:)-hog_gauss_data.sampleMean;
    lambda_ = 0.01;
    x_1 = (hog_gauss_data.d./(hog_gauss_data.d.^2+lambda_.^2)).*x_1;
    X_w{k} = x_1(:);
    %     clf; subplot(1,3,1); imagesc(curImage);axis image;
    %     subplot(1,3,2); imagesc(showHOG(conf,x));axis image;
    %     subplot(1,3,3); imagesc(showHOG(conf,x_1));axis image;
    %clf; imshow(curImage);
    %     pause;
end


%%


%% mark the top of each cup.

%multiWrite(L.cupImages,'cupImages');

bbLabeler({'cup'},'cupImages','cupImages/anno');


d = dir(fullfile('cupImages','*.jpg'));
for k = 1:length(d)
    resNm = fullfile('cupImages/anno',[d(k).name,'.txt']);
    objs=bbGt('bbLoad',resNm);
    
    clf; imagesc(imread(fullfile('cupImages',d(k).name))); axis image;
    hold on; plotBoxes2(objs.bb([2 1 4 3]),'g');
    pause;
end

[rects] = selectSamples(conf,cupImages,'cupImagesAnno')

rects1 = cat(1,rects{:});
R = multiCrop(conf,cupImages,imrect2rect(round(rects1)),[30 50]);
mImage(R);
conf.features.vlfeat.cellsize = 4;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
[X,sz] = imageSetFeatures2(conf,R,true,[30 50]);
[IDX,C] = kmeans2(X',5,struct('nTrial',100));
%       clusters = makeClusters(X,[]);
[clusters,ims] = makeClusterImages(mat2cell2(imgs,[1 1 1 size(IJ,4)]),C',IDX',X,'dir1');

save clusters clusters
sz = sz{1}
save clusterData clusters_trained sz
conf.features.winsize = sz{1};
% clusters = makeCluster(X,[]);
clusters_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','cups_1','override',true);
%clusters_trained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','cups_1','override',false);
figure,imshow(showHOG(conf,clusters_trained(5).w))
imshow(multiImage(R,true,false));

conf.get_full_image = false;
qq = applyToSet(conf,clusters_trained,test_ids(test_labels),[],'cup_top_check','override',true,'disp_model',true,...
    'uniqueImages',false);

%
% qq1 = qq(1);

conf.class_subset = conf.class_enum.DRINKING;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
TT = test_ids(test_labels);
imshow(getImage(conf,TT{1}))
conf.get_full_image = false;
conf.max_image_size = 400;
%%
% X = cat(2,X_w{1:20});
% conf.features.vlfeat.cellsize = 8;
% conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
%
% clusters_trained = makeClusters(double(X),[]);
% conf.detection.params.max_models_before_block_method = 100;
% % clusters_trained = makeClusters(X,[]);
conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_min_scale = .1;
for k = length(TT):-1:1
    I = getImage(conf,TT{k});
    conf.detection.params.detect_exemplar_nms_os_threshold = 0;%.8;%.1;
    %     I = imcrop(I);
    %     I = imrotate(I,40,'bilinear','crop');
    qq = applyToSet(conf,clusters_trained,{I},[],'cup_top_check','override',true,'disp_model',true,...
        'uniqueImages',false);
    %     iq = 2;
    Z = zeros(dsize(I,1:2));
    for iq = 1:length(qq)
        iq
        cluster_locs = cat(1,qq(iq).cluster_locs);
        B = [round(cluster_locs(:,[1:4])) cluster_locs(:,12)];
        B = clip_to_image(B,[1 1 dsize(I,[2 1])]);
        clf; subplot(2,1,1); imagesc(I); axis image;
        m = min(B(:,end));
        %     B(:,end) = B(:,end)-min(B(:,end));
        %     B = B(1:min(1000,size(B,1)),:);
        %     hold on; plotBoxes2(B(:,[2 1 4 3]));
        %     M = computeHeatMap(I,B,'max');
        bc = round(boxCenters(B));
        M = m*ones(dsize(I,1:2));
        for b = 1:size(bc,1)
            M(bc(b,2),bc(b,1)) = max(M(bc(b,2),bc(b,1)),B(b,end));
        end
        %     M = imfilter(M,fspecial('gauss',9,3),m);
        M = imdilate(M,strel('disk',3));
        subplot(2,1,2); imagesc(M);axis image;colorbar
        %         Z = Z+normalise(M);
        %         subplot(1,3,3); imagesc(Z); axis image;
        pause;
    end
end
%%
imshow('/home/amirro/code/mircs/cup_top_check.jpg');





for iImage = 1:length(L.cupImages)
    iImage
    curImage = L.cupImages{iImage};
    clf; imagesc(curImage);
    [h,api]=imRectRot;
end
%[hPatch,api] = imRectRot( cupImages{1} )


%%
initpath;
config;
%%
baseDir = '~/storage/dpm_drink_s40';
conf.class_subset = conf.class_enum.JUMPING;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
addpath('/home/amirro/code/3rdparty/objectness-release-v2.0/');

%% show some objectness results
baseDir = '~/storage/lineseg_s40';
% test_images  = test_ids;
test_images = test_ids(test_labels);

ir = 1:length(test_images);
% ir = randperm(length(test_images));
% ir = iv
conf.get_full_image = true;
max_image_size = inf;
for q = 1:length(test_images)
    k = ir(q);
    currentID = test_images{k};
    % currentID = 'drinking_008.jpg';
    I = getImage(conf,currentID);
    ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
    load(ucmFile); % ucm
    propsFile = fullfile('~/storage/geometry_s40',strrep(currentID,'.jpg','.mat'));
    load(propsFile);
    segmentBoxes = cat(1,props.BoundingBox);
    segmentBoxes = imrect2rect(segmentBoxes);
    B = load(fullfile('~/storage/objectness_s40',strrep(currentID,'.jpg','.mat')));
    
    clf;
    m = 2; n =3 ;
    subplot(m,n,1); imagesc(I); axis image;
    subplot(m,n,2); imagesc(ucm); axis image;
    load(fullfile(baseDir,strrep(currentID,'.jpg','.mat')));
    params.windows = segmentBoxes;
    %     objScores = runObjectness(I,10000,params);
    
    subplot(m,n,5);
    imagesc(I); axis image;hold on;
    
    %     [edgelist,seglist] = processEdges(ucm>.1);
    
    drawedgelist(seglist,dsize(I,1:2),1,'rand');
    doObjectness = true;
    if (doObjectness)
        [map,counts] = computeHeatMap(I,B.boxes(1:500,:),'sum');
        [regions,regionOvp,G] = getRegions(conf,currentID,false);
        [subsets] = suppresRegions(regionOvp,.1);
        regions = regions(subsets{1});
        map2 = zeros(size(map));
        for rr = 1:length(regions)
            rr
            %map = map+regions{rr}*objectnessScores(rr);
            map2 = map2 + regions{rr}*mean(map(regions{rr}));
            %%%*(1./(1+.1*length(regions)-rr));
        end
        subplot(m,n,3); imagesc(map); axis image;
        subplot(m,n,4); imagesc(map2); axis image;
    end
    pause;
    
end
%%

imgDir = '/home/amirro/storage/data/Stanford40/JPEGImages';
addpath('~/code/utils');
% L = load('~/code/mircs/dpm_models/cup');
% cd /home/amirro/code/3rdparty/voc-release5;
% startup;
% model = L.partModelsDPM_cup{1};

load cupImages;
mImage(cupImages)
%%
ir = permute
for q = 1:length(test_images)
    
    I = getImage(conf,test_images{k});
    %I = imread(fullfile(imgDir,test_images{k}));
    %
    % %     imshow(I)
    %     I1 = imresize(I,[32 NaN]);
    %     I1 = imresize(I1,[128 NaN]);
    %     clf; imagesc(I1); axis image;
    %     pause;
    % end
    %%
    
    %     visualizemodel(model)
    
    load(fullfile(baseDir,strrep(test_images{k},'.jpg','.mat')));
    
    boxes = shape(2).boxes;
    
    [s,is] = sort(boxes(:,end),'descend');
    boxes = boxes(is(1:100),:);
    
    clf;
    subplot(1,2,1); imagesc(I); axis image; hold on;
    hold on; plotBoxes2(boxes(:,[2 1 4 3]));
    boxes = clip_to_image(boxes,[1 1 dsize(I,[2 1])]);
    %
    [map,counts] = computeHeatMap(I,boxes(:,[1:4 6]),'sum');
    subplot(1,2,2); imagesc(map./counts); axis image;
    pause;
    
end

%%
for k = 1:length(test_images)
    k
    I = getImage(conf,test_images{k});
    load(fullfile(baseDir,strrep(test_images{k},'.jpg','.mat')));
    
    boxes = shape(1).boxes;
    
    [s,is] = sort(boxes(:,end),'descend');
    boxes = boxes(is(1:10),:);
    
    clf;
    subplot(1,2,1); imagesc(I); axis image; hold on;
    hold on; plotBoxes2(boxes(:,[2 1 4 3]));
    boxes = clip_to_image(boxes,[1 1 dsize(I,[2 1])]);
    %
    [map,counts] = computeHeatMap(I,boxes(:,[1:4 6]),'sum');
    subplot(1,2,2); imagesc(map./counts); axis image;
    pause;
    
end
%%



rprops = regionprops(e1,'PixelList','Area','PixelIdxList');
areas = [rprops.Area];

rprops = rprops(areas >= 10);
Z = zeros(size(e1));
for k = 1:length(rprops)
    Z(rprops(k).PixelIdxList) = k;
    cur_ellipse =  fit_ellipse(rprops(k).PixelList(:,1),rprops(k).PixelList(:,2));
    
    rprops(k).curvature = LineCurvature2D(rprops(k).PixelList);
    Z(rprops(k).PixelIdxList) = rprops(k).curvature;
    
    rprops(k).ellipse = cur_ellipse;
    if (isempty(cur_ellipse) || isempty(cur_ellipse.a))
        continue;
    end
end

clf,
subplot(2,2,1),imshow(Z,[]); hold on;
for k = 1:length(rprops)
    cur_ellipse  = rprops(k).ellipse;
    if (isempty(cur_ellipse) || isempty(cur_ellipse.a))
        continue;
    end
    pts = ellipse_points(cur_ellipse);
    plot(pts(1,:),pts(2,:),'r','LineWidth',2);
end
subplot(2,2,2),imshow(e1,[]);
subplot(2,2,3),imshow(c1);

%segments = RemapLabels(vl_slic(im2single(vl_xyz2lab(vl_rgb2xyz(c1))),100,.0001));
%segments = RemapLabels(vl_slic(im2single(vl_xyz2lab(vl_rgb2xyz(c1))),200,.0001));
segments = RemapLabels(vl_slic(im2single(c1),100,.01));

[segImage,c] = paintSeg(c1,segments);

subplot(2,2,4); imshow(segImage);
%imshow(edge(double(segments),'canny'),[])


%        [R,F] = vl_mser(im2uint8(rgb2gray(c1)));
%        F = vl_ertr(F);
%
%        subplot(2,2,4); imshow(c1); hold on;
%        vl_plotframe(F);
pause;

%%%%% old sequential detection
%%
E = edge(curIm_gray,'canny');
[edgeList,labeledEdgeIm] = edgelink(E);
tol = 2;
seglist = lineseg(edgeList, tol);
% search for edges in area of interest
segs = seglist2segs(seglist);

% display stuff
figure(1);
clf;
subplot(mm,nn,1);
imagesc(curIm); axis image; title('orig');
subplot(mm,nn,2);
imagesc(labeledEdgeIm); axis image; title('labeled edges');
subplot(mm,nn,3);
drawedgelist(seglist, size(curIm), 2, 'rand'); title('seglist');
subplot(mm,nn,4);
imagesc(Z); axis image;
drawedgelist(seglist, size(curIm), 2, 'rand'); title('seglist with roi');


% flip so "starting point" is above end point.
yStart = segs(:,1);
yEnd = segs(:,3);
toFlip = yStart > yEnd;
segs(toFlip,:) = segs(toFlip,[3 4 1 2]);

xStart = segs(:,2);
xEnd = segs(:,4);
yStart = segs(:,1);
yEnd = segs(:,3);
% check which segments are in the correct region
sel_region = Z(sub2ind(size(Z),yStart,xStart));

% check which segment line are parallel..
vecs = segs2vecs(segs);
[X,norms] = normalize_vec(vecs');
cos_angles = X'*X;
% remove self-angle
cos_angles = cos_angles.*(1-eye(size(cos_angles)));
maxAngle = 5; % maximal angle between adjacent segments.
[ii,jj] = find(abs(cos_angles) >= cosd(maxAngle)); % ii,jj are possible pairs of segments.
%     figure,imagesc(real(acosd(cos_angles)))

maxDistance = 20*scaleFactor; % maximal distance between starting points
dists = sum((segs(ii,1:2)-segs(jj,1:2)).^2,2).^.5;
dist_sel = dists <= maxDistance;
ii = ii(dist_sel);
jj = jj(dist_sel);
dists = dists(dist_sel);
in_region = sel_region(ii) & sel_region(jj);
lengths = max(norms(ii),norms(jj));
T_length = 10*scaleFactor;
in_region = in_region & (lengths' >= T_length);
ii = ii(in_region);
jj = jj(in_region);
dists = dists(in_region);
%
segCenters = (segs(:,1:2)+segs(:,3:4))/2;

subplot(mm,nn,5) ;
imagesc(curIm); axis image;hold on;
for n = 1:size(ii,1)
    line(segs(ii(n),[2 4]), segs(ii(n),[1 3]),...
        'LineWidth', 1, 'Color', 'r');
    line(segs(jj(n),[2 4]), segs(jj(n),[1 3]),...
        'LineWidth', 1, 'Color', 'r');
    
    plot(segCenters([ii(n) jj(n)],2), segCenters([ii(n) jj(n)],1),'--m');
end

segments = vl_slic(single(vl_xyz2lab(vl_rgb2xyz(im2single(curIm)))),scaleFactor*10,.00001);
[segImage,c] = paintSeg(curIm,segments);
subplot(mm,nn,6);
%     imagesc(rgb2gray(segImage));colormap gray; axis image;
imagesc((segImage)); ; axis image;




%     curIm = cropper(train_faces{k},curBox);
% % %     gPb_orient = cropper(gPb_orient,curBox);
% % %     gPb_thin = cropper(gPb_thin,curBox);
% % %
% % %     % gbp!
% % %     [gPb_orient, gPb_thin, textons] = globalPb(curIm);


%%

% obtain the mouth images...
imageSet = imageData.train;
sz = [64 64];
% get just the top 1000 scoring faces.
scores = imageSet.faceScores;
[s,is] = sort(scores,'descend');
top_k = 1000;
is = is(1:min(length(is),top_k));
subs = zeros([sz 3 length(is)],'uint8');
for k = 1:length(is)
    imageInd = is(k);
    k
    currentID = imageSet.imageIDs{imageInd};
    I = getImage(conf,currentID);
    faceBoxShifted = imageSet.faceBoxes(imageInd,:);
    %         lipRectShifted = imageSet.lipBoxes(imageInd,:);
    %         box_c = round(boxCenters(lipRectShifted));
    % get the radius using the face box.
    %         [r c] = BoxSize(faceBoxShifted);
    %         boxRad = (r+c)/2;
    %         bbox = [box_c(1)-r/4,...
    %             box_c(2),...
    %             box_c(1)+r/4,...
    %             box_c(2)+boxRad/2];
    %         bbox = round(bbox);
    
    %         if (any(~inImageBounds(size(I),box2Pts(bbox))))
    %             continue;
    %         end
    bbox = round(faceBoxShifted);
    I_sub = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    I_sub = imResample(I_sub,sz,'bilinear');
    %         I_sub = rgb2gray(I_sub);
    subs(:,:,:,k) = im2uint8(I_sub);
end

%     montage2((subs),struct('hasChn',1))
%%
M = mat2cell2(im2single(subs),[1 1 1 size(subs,4)]);
conf.features.vlfeat.cellsize = 8;
X_faces = imageSetFeatures2(conf,M,true,sz);
%%


% now, for a new test image, find the best similarity.....
%%
conf.get_full_image = false;
for k = 6:length(test_images)
    
    I = getImage(conf,test_images{k});
    %         I = getImage(conf,'drinking_001.jpg');
    
    % find the best distance....
    [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,I,.5 );
    
    
    D = l2(X',X_faces');
    DD = exp(-D/100);
    boxes = clip_to_image(round(boxes),[1 1 dsize(I,[2 1])]);
    bb = [boxes(:,1:4),sum(DD,2)];
    
    [b,ib] = sort(bb(:,end),'descend');
    
    H = computeHeatMap(I,bb(ib(1:100),:),'max');
    clf;
    subplot(2,1,1); imagesc(I); axis image;
    subplot(2,1,2); imagesc(H); axis image;
    pause;
    
end
%%


% load upper body detections for entire set...
for k = 1:length(newImageData)
    k
    resPath = j2m(conf.upperBodyDir,newImageData(k).imageID);
    load(resPath);
    if (~isempty(res))
        res = res(:,[1:4 6]);
    else
        res = [];
    end
    newImageData(k).upperBodyDets = res;
end
%%

scores = {};
subs = {};
%%

for k = 1:length(newImageData)
    k
    newImageData(k).upperBodyDets = getUpperBodyDets(conf,newImageData(k));
end

%%
for k = 1:length(subs)
    
    M = imResample(subs{k},[300 300],'bilinear');
    for rot = -30:10:30
        M_rot = imrotate(M,rot,'bilinear','crop');
        bbs = acfDetect(M_rot,detectors);
        %     bbs = cat(1,bbs{:});
        if (~isempty(bbs))
            curScore = max(curScore,max(bbs(:,5)));
            [b,ib] = sort(bbs(:,5),1,'descend');bbs = bbs(ib(1),:);
            %                 bbs
            bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
            clf; imagesc(M_rot); axis image; hold on; plotBoxes(bbs,'g','LineWidth',2);drawnow; pause(.1)
        end
    end
end


%% show piotr's face detection results.
for k = 1:length(newImageData)
    curImageData = newImageData(k);
    if (~curImageData.label),continue,end;
    L_bb = load(j2m('~/storage/s40_piotr_faces_big_full_detection/',curImageData));
    I = getImage(conf,curImageData);
    bbs = L_bb.bbs;
    %bbs = bbs(bbs(:,5)>30,:);
    [r,ir] = sort(bbs(:,5),'descend'); bbs = bbs(ir(1:min(size(bbs,1),5)),:);
    bbs = rotate_bbs(bbs,I,bbs(:,end));
    clf;   imagesc(I); axis image; hold on; plotPolygons(bbs(:),'g');drawnow; pause;
end

%% some symmetry features
%imagesc(face_images{2})
% load ~/storage/misc/face_images.mat;
% I = face_images{2};
% cd /home/amirro/code/3rdparty/symmetry_1.0/

imgData = newImageData(img_sel);
load ~/code/mircs/faceActionImageNames;
%%
for t = 1:length(newImageData)
    curImageData = newImageData(t);
    
    [I,I_rect] = getImage(conf,curImageData.imageID);
    %     upperBodyDets = curImageData.upperBodyDets;
    if (~exist(j2m('~/storage/s40_heads',curImageData.imageID),'file'))
        continue;
    end
    
    headDets = load(j2m('~/storage/s40_heads',curImageData.imageID));
    headDets = headDets.res;
    [ovp,ints] = boxesOverlap(headDets,I_rect);
    [~,~,areas] = BoxSize(headDets);
    %     headDets(ints./areas < .6,:) = [];
    %     ints = bsxfun(@rdivide,ints,areas);
    %     [s1,s2] = meshgrid(upperBodyDets(:,end),headDets(:,end));
    %     totalScore = s1+s2;
    %     totalScore(ints<.8) = -infu;
    %     [m,im] = max(totalScore(:));
    %     [iHead,~] = ind2sub(size(totalScore),im);
    headDets = headDets(1:min(10,size(headDets,1)),:);
    
    
    colors = hsv(10);
    
    clf; imagesc2(I); hold on;
    for tt = 1:min(3,size(headDets,1))
        curCenter = boxCenters(headDets(tt,:));
        
        plotBoxes(headDets(tt,:),'color',colors(tt,:),'LineWidth',2);
        showCoords(curCenter,{[num2str(tt), ', ', num2str(headDets(tt,5))]});
        
    end
    %plotBoxes(headDets(iHead,:),'g--','LineWidth',2);
    %plotBoxes(headDets(1,:),'r--','LineWidth',2);
    pause;
end



%% foreground saliency maps...
load ~/code/mircs/faceActionImageNames;

%%
T = .85;
zzz = 'drink';
for t = 1:length(newImageData)
    t
    clc
    currentID = newImageData(t).imageID;
    if (~strncmp(currentID,zzz,length(zzz)))
        continue
    end
    imgData = newImageData(findImageIndex(newImageData,currentID));
    [I,I_rect] = getImage(conf,currentID);%I=imresize(I,2);I = clip_to_bounds(I);
    L = load(j2m('~/storage/s40_sal_fine',currentID));
    res = L.res(1);
    res = (1-res.sal_bd+res.sal)/2;
    %     res = L.res;
    %     res = normalise(res);
    clf;
    vl_tightsubplot(2,2,2); imagesc2(res);
    %     res = imResample(res,200/size(res,1));
    %     I = imResample(I,size(res));
    res(res < .1) = 0;
    res = addBorder(res,1,0);
    %     [seg_mask,energies] = st_segment(im2uint8(I),res,.7,10,5);
    %     res(res < T) = 0;
    %     break;
    
    rr = poly2mask2(rect2pts(I_rect),size2(I));
    %     res = imResample(single(res),size2(I));
    sz = size2(I);
    sizeRatio = size(res,1)/size(I,1);
    I = imResample(I,size(res));
    rr = imResample(single(rr),size(res));
    %     res = double(resaliencys.*rr);
    
    headDets = load(j2m('~/storage/s40_heads',currentID));
    headDets = headDets.res;
    %     headDets(:,1:4) = bsxfun(@plus,headDets(:,1:4),I_rect([1 2 1 2]));
    headDets = headDets(1:min(3,size(headDets,1)),:);
    headDets = headDets*size(res,1)/sz(1);
    
    
    vl_tightsubplot(2,2,1);
    imagesc2(I);
    %     vl_tightsubplot(2,2,2); imagesc2(res);
    q = blendRegion(I,double(res)>T,1);
    vl_tightsubplot(2,2,3); imagesc2(q);
    %     vl_tightsubplot(2,2,4); displayRegions(I,seg_mask);
    
    %     pause;continue
    
    %     colorspecs = {'g--','r--','b--'};
    %     for t = 1:min(3,size(headDets,1))
    %         hold on;
    %         plotBoxes(headDets(t,:),colorspecs{t},'LineWidth',2);
    %
    %     end
    
    %
    %     n = imgData.upperBodyDets;
    %     plotBoxes(n(1:min(3,size(n,1)),:)*size(I,1)/sz(1),'m-','LineWidth',2);
    %     plotBoxes(I_rect*sizeRatio);
    res = addBorder(res,1,0);
    
    
    
    %     killBorders = 0; %useful : zero out probabilities of superpixels within 1 pixels of image borders.
    %     segs = vl_slic(im2single(I),.1*size(I,1),1);
    %     segs = RemapLabels(segs); % fix labels to range from 1 to n, otherwise a mex within constructGraph crashes.
    %     graphData = constructGraph(I,res,segs,killBorders);
    %
    %     edgeParam = .01;  % A very important number!!!! must be > 0
    %     [labels,labelImage] = applyGraphcut(segs,graphData,edgeParam);
    % display results...
    %     clf;
    %     subplot(2,2,1); imagesc(I); axis image; hold on; plotBoxes2(bbox(:,[2 1 4 3]),'g');
    %     subplot(2,2,2); imagesc(pMap); axis image; title('unary factor');hold on; plotBoxes2(bbox(:,[2 1 4 3]),'g');
    %     subplot(2,2,4); imagesc(graphData.seg_probImage); axis image; title('unary factors (superpix. averaging)');
    %     subplot(2,2,3); imagesc(labelImage); axis image; title('mrf result');
    %     displayRegions(I,{labelImage},0,-1)
    %
    
    %     displayRegions(I,seg_mask);
    %     pause;continue;
    
    
    drawnow;
    pause;
end
%%

addpath('/home/amirro/code/3rdparty/bcp_release/learning');
addpath('/home/amirro/code/3rdparty/bcp_release/data_management/');
addpath('/home/amirro/code/3rdparty/bcp_release/candidates/quantombone-exemplarsvm-850fcb0/features/');
rmpath(genpath('/home/amirro/code/3rdparty/face-release1.0-basic/'));
% bbox = [1 1 fliplr(size2(I))];
%%
clf;imagesc2(I);
[~,api]=imRectRot('rotate',0);
objs = bbGt( 'create', 1 );
objs.lbl = 'face';
bbox = round(api.getPos());bbox(3:4) = bbox(3:4)+bbox(1:2);
models_name = sprintf('exemplar-lda-%s-%d', mat2str(bbox), MAXDIM);
MAXDIM = 10;
params.sbin = 8;
params.interval = 10;
params.MAXDIM = MAXDIM;

model = initialize_goalsize_model(convert_to_I(I), bbox, params);
model = train_whog_exemplar(model);
model.name = models_name;
model = convert_part_model(model);
im = HOGpicture_qb(model.filter, 15);
figure,imagesc2(im);
figure,imagesc2(cropper(I,round(model.bb)));

%% clean up hands directory...
%
% baseDir = '~/storage/hands_s40';
% d = dir(fullfile(baseDir,'*.mat'));
%
% for t = 1:length(d)
%     t
%     f = fullfile(baseDir,d(t).name);
%     load(f);
%
%     shape = rmfield(shape,'bboxes');
%     shape = rmfield(shape,'boxes_r');
%     context = rmfield(context,'bboxes');
%     context = rmfield(context,'boxes_r');
%     save(f,'shape','context');
% end

%% person detection grammar.
load ~/code/mircs/faceActionImageNames;
zzz ='drink';
clf;imagesc(I);axis image;
% ds = imgdetect(I,model,0);
% pick = nms(ds,.1);
showboxes(I,ds(pick,:));

%% person detection results.

%%
zzz = 'blow';

for t = 1:length(faceActionImageNames)
    currentID = faceActionImageNames{t};
    [~,currentID,ext] = fileparts(currentID);currentID = [currentID ext];
    currentID
    if (~strncmpi(currentID,zzz,length(zzz)))
        continue
    end
    [I,I_rect] = getImage(conf,currentID);%I=imresize(I,2);I = clip_to_bounds(I);
    %     clf;imagesc2(I); break;
    L = load(j2m('/home/amirro/storage/s40_voc5_person',currentID));
    res = L.res;
    I = padarray(I,[100 100],0,'both');
    clf;
    imagesc2(I);
    if (~isempty(res))
        res(:,1:4)=res(:,1:4)+100;
        plotBoxes(res);
    else
        title('no boxes');
    end
    
    
    
    
    
    drawnow;pause;continue;
end

%% some poselets
load ~/code/mircs/faceActionImageNames;
zzz ='drink';
clf;imagesc(I);axis image;
% ds = imgdetect(I,model,0);
% pick = nms(ds,.1);
showboxes(I,ds(pick,:));

%% person detection results.

%%
zzz = 'smok';

for t = 1:length(faceActionImageNames)
    currentID = faceActionImageNames{t};
    [~,currentID,ext] = fileparts(currentID);currentID = [currentID ext];
    currentID
    if (~strncmpi(currentID,zzz,length(zzz)))
        continue
    end
    [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);%I=imresize(I,2);I = clip_to_bounds(I);
    [rects,rects_poselets,poselet_centers,poselet_ids,s,is] = ...
        getPoseletData(conf,currentID,xmin,ymin,xmax,ymax);
    [map,counts] = computeHeatMap(I,rects,'sum');
    map = map/max(map(:));
    figure(1);clf;imagesc2(sc(cat(3,map,I),'prob'));%  plotBoxes(rects);
    %     figure(2);clf; imagesc2(map);
    drawnow;pause;continue;
end

%% upper body.
zzz = 'dri';

for t = 1:length(fra_db)
    currentID = fra_db(t).im
    %currentID = faceActionImageNames{t};
    [~,currentID,ext] = fileparts(currentID);currentID = [currentID ext];
    currentID
    if (~strncmpi(currentID,zzz,length(zzz)))
        continue
    end
    imgData = newImageData(findImageIndex(newImageData,currentID));
    [I,I_rect] = getImage(conf,currentID);%I=imresize(I,2);I = clip_to_bounds(I);
    
    clf; imagesc(I); hold on;
    s = computeHeatMap(I,imgData.upperBodyDets,'sum');
    
    II = sc(cat(3,s,I),'prob');
    clf; imagesc2(II);
    
    drawnow;pause;continue;
end


%% upper body - new version...
zzz = 'look';
faceActionImageNames = s40_fra;

for t = 1:length(faceActionImageNames)
    currentID = faceActionImageNames(t).imageID;
    [~,currentID,ext] = fileparts(currentID);currentID = [currentID ext];
    currentID
    if (~strncmpi(currentID,zzz,length(zzz)))
        continue
    end
    imgData = faceActionImageNames(t);
    [I,I_rect] = getImage(conf,currentID);%I=imresize(I,2);I = clip_to_bounds(I);
    load(j2m('/home/amirro/storage/s40_upper_body_2',currentID));
    %     clf; imagesc(I); hold on;
    %     s = computeHeatMap(I,imgData.upperBodyDets,'sum');
    clf; imagesc2(I); hold on;
    plotBoxes(boxes(1:min(1,size(boxes,1)),:));
    %     II = sc(cat(3,s,I),'prob');
    %clf; imagessc2(II);
    drawnow;pause;continue;
end

%% uiuc upper body
d = dir('~/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages/*.jpg');
%%
crops = {};
scores = {};
for t = 1:length(d)
    t
    fName = fullfile('~/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages/',...
        d(t).name);
    L =load(j2m('/home/amirro/storage/uiuc_upper_body',fName));
    res = L.res;
    if (~isempty(res))
        I = imread(fName);
        crops{end+1} = cropper(I,round(res(1,1:4)));
        scores{end+1} = res(1,5);
    end
    %     clf;imagesc2();
    
    %hold on; plotBoxes(L.res);
    %     pause;
end
scores = cat(1,scores{:});
showSorted(crops,scores,1000);


%% new k-poselets...
bd = '/home/amirro/storage/s40_k_poselets';
f = dir(fullfile(bd,'*.mat'));
addpath(genpath('/home/amirro/code/3rdparty/k-poselets-master/code'));
%%
for k = 500:length(f)
    [~,name,ext] = fileparts(f(k).name);
    I = getImage(conf,[name '.jpg']);
    %     clf;
    %     imagesc(I);
    L = load(fullfile(bd,f(k).name));
    showkeypoints(I,L.res.kp_coords);
    pause
end



%%%%%
addpath('/home/amirro/code/3rdparty/smallcode/');
%
% X = rand(10000,800);
% [recall, precision] = test(X, 256, 'ITQ');
%%
default_init;
specific_classes_init;
cls = 1;
all_opt_params = {};
ppp = randperm(length(classes));
for iClass = 1:length(classes)
    %     iClass=ppp(pClass);
    cls = iClass;
    resPath = fullfile('~/storage/misc',[classNames{iClass} '_params.mat']);
    if (exist(resPath,'file'))
        continue;
    end
    save(resPath ,'cls');
    cls
    sel_train = class_labels ==cls & isTrain;
    % parameters
    params.img_h = [100 150 200];
    params.wSize = [2 4 6];
    params.nIter = 100;%70;
    params.min_scale = 1;%[.7 1];
    params.useSaliency = [0 1];
    params.nn = [100 500]; %params.nn = [50 100 500];
    optParams = findOptParams(conf,newImageData(validIndices),params,sel_train);
    save(resPath,'optParams');
end
% save all_opt_params all_opt_params;

%% check dpm results....

load fra_db.mat;

load dpm_models/fra_models.mat;
%{all_models.class}' % drinking head is 3.
model_res = {};
scores = -inf(size(fra_db));

all_res = {};
all_imgs = {};
for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);
    if (curImgData.isTrain),continue;end
    all_res{t} = load(j2m('~/storage/s40_fra_dpm',curImgData.imageID));
    %     faceBox = inflatebbox(curImgData.faceBox,2.5,'both',false);
    %     I = getImage(conf,curImgData);
    %     I = cropper(I,round(faceBox));
    %     all_imgs{t} = im2uint8(I);
    %     scores(t) = L.res(3).boxes(1,5);
end

%%
% hand, obj,head,phrase
iClass = 5; % drink % drink, smoke,blowing, brushing,phone
scores = -inf(size(fra_db));
for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);
    if (curImgData.isTrain),continue;end%
    if (curImgData.classID~=iClass),continue,end;
    [~,~,I] = get_rois_fra(conf,curImgData);
    
    z = (iClass-1)*4;
    scores(t) = 0;
    clf;
    %     faceBoxExt = inflatebbox(curImgData.faceBox,2.5,'both',false);
    %       close all
    curRes = all_res{t}.res;
    %       I = getImage(conf,curImgData);
    %       I = cropper(I,round(faceBoxExt));
    for q = [1]
        rr = curRes.detections(z+q);
        bb = rr.boxes;
        clf;imagesc2(I);hold on;plotBoxes(bb(1:min(1,size(bb,1)),:));
        pause
        tt = -1;
        if (~isempty(bb))
            tt = bb(1,5);
        end
        scores(t) = scores(t)+tt;
    end
end
showSorted(all_imgs,scores,20);

%% check the star model.
data_path = '/home/ita/code/ship_MIRC_points/points';
addpath('~/code/3rdparty/annstar')
annotfile = 'Horse17_data.mat';
load(fullfile(data_path, annotfile));
[NG,PS,PSTR] = convert_data_base_format_new(17,{[4 1]},[],0);
train_imgs_30x30 = PSTR;
test_imgs_30x30 = PS{1}{1};
[params, sms, train_posXY, encoders] = my_star_model_train(train_imgs_30x30, train_annotations, true);
[test_estimatePosXY, test_support_map, res_data] = my_star_model_test(test_imgs_30x30, params, sms, train_posXY, encoders, train_imgs_30x30, true);
%%
% new segmentations stuff....
load fra_db.mat;

for t = 1:length(fra_db)
    t
    curImgData = fra_db(t);
    [~,roiBox,I] = get_rois_fra(conf,curImgData);
    load(j2m('~/storage/s40_seg_new',curImgData));
    candidates = res.cadidates;
    masks = {};
    
    % retain only the superpixels within the head area
    candidates.superpixels = cropper(candidates.superpixels,roiBox);
    candidates.superpixels = imResample(candidates.superpixels,size2(I),'nearest');
    uSuperPix = unique(candidates.superpixels(:));
    goods = cellfun(@(x) any(ismember(x,uSuperPix)),candidates.labels);
    candidates.labels = candidates.labels(goods);
    % now get unique groups...
    lengths = cellfun(@length,candidates.labels);
    uLengths = unique(lengths);
    newLabels = {};
    for tt = 1:length(uLengths)
        curGroups = lengths == uLengths(tt);
        V = unique(cat(1,candidates.labels{curGroups}),'rows');
        
    end
    for kk = 1:length(candidates.labels)
        kk
        masks{kk} = ismember(candidates.superpixels, candidates.labels{kk});
    end
    %     figure,imagesc2(I);
    %     figure,imagesc2( candidates.superpixels);
    
    %     I =getImage(conf,curImgData);
    displayRegions(I,masks,[],.1);
    
    [regions,ovp] = chooseRegion(I,masks,.5);
    displayRegions(I,regions,ovp,0);
    if (curImgData.isTrain),continue;end
    all_res{t} = load(j2m('~/storage/s40_fra_dpm',curImgData.imageID));
    
    
    %     faceBox = inflatebbox(curImgData.faceBox,2.5,'both',false);
    %     I = getImage(conf,curImgData);
    %     I = cropper(I,round(faceBox));
    %     all_imgs{t} = im2uint8(I);
    %     scores(t) = L.res(3).boxes(1,5);
end


%% some more saliency experiments...
initpath;
config;
addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
load fra_db.mat

%% 2. Saliency Map Calculation
%%
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 25;
opts.pixNumInSP = spSize;
conf.get_full_image = true;
for t = 200:10:length(fra_db)
    [rois,bbox,I] = get_rois_fra(conf,fra_db(t),2,128);
    
    [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I),opts);
    n = n+1;
    res(n).sal = sal;
    res(n).sal_bd = sal_bd;
    res(n).resizeRatio = resizeRatio;
    res(n).bbox = bbox;
    clf,subplot(1,2,1);imagesc2(I);
    subplot(1,2,2),imagesc2(sal);
    drawnow;pause
end

%% show some performance for my ann-star
infScale = 2.5;
absScale = 200;
iClass = 1;
for k = 1:length(fra_db)
    k
    curImageData = fra_db(k);
    if (curImageData.classID~=iClass)
        continue;
    end
    if (curImageData.isTrain),continue,end;
    [rois,subRect,I] = get_rois_fra(conf,curImageData,infScale);
    boxes = zeros(length(classNames),4);
    rois = {};
    
    %     for iClass = 1:length(classNames)
    
    probPath = fullfile('~/storage/s40_fra_box_pred',[curImageData.imageID '_' classNames{iClass} '.mat']);
    
    load(probPath);
    pMap = imResample(pMap,size2(I));
    pMap = normalise(pMap);
    clf; imagesc2(sc(cat(3,pMap,I),'prob'));title(classNames{iClass});
    pause;
    rois{iClass} = pMap > .5;
    %     end
end


%%
p = randperm(length(fra_db));
for ik=1:length(fra_db)
    k = p(ik);
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k));
    mouthBox = rois(end).bbox;
    I = cropper(I,round(mouthBox));
    clf; imagesc2(I); pause;
end


%%


% ratios =
obj_sums = zeros(length(fra_db),length(objTypes),2);
counts = zeros(length(fra_db),length(objTypes));



% statistics of object saliency inside / outside object pixels

for t = 1:length(fra_db)
    t
    if (~isTrain(t)),continue,end;
    %     if (fra_db(t).classID~=5),continue;end
    salMap = foregroundSaliency(conf,fra_db(t).imageID);
    roiParams.absScale = -1;
    [rois,roiBox,I] = get_rois_fra(conf,fra_db(t),roiParams);
    salMap = cropper(salMap,roiBox);
    for iRoi = 1:length(rois)
        %         iRoi
        objType = rois(iRoi).id;
        if (objType == 5),continue,end;
        curBox = clip_to_image(round(rois(iRoi).bbox),I);
        objMask = poly2mask2(curBox,size2(I));
        counts(t,objType) = counts(t,objType)+1;
        obj_sums(t,objType,1) = obj_sums(t,objType,1)+mean(salMap(objMask));
        obj_sums(t,objType,2) = obj_sums(t,objType,2)+mean(salMap(~objMask));
    end
    %     clf; subplot(1,2,1);imagesc2(I);
    %     subplot(1,2,2);imagesc2(salMap>.1);colorbar; drawnow;
    %     pause
end

mean_ratios = bsxfun(@rdivide,obj_sums(:,:,1),counts)./bsxfun(@rdivide,obj_sums(:,:,2),counts);
m = mean_ratios(:,3);
m(isnan(m)) = [];

[no,xo]=hist(m,100);
pd = fitdist(m,'rayl');
ff = pd.pdf(xo);
ff = ff/sum(ff);
figure,plot(xo,ff);hold on; plot(xo,no/sum(no),'r');
%% can check how many times you "hit" the action object with the top k proposals.

%  addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV/'));

for t = 1:length(fra_db)
    [rois,~,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t));
    rois = rois([rois.id]==3);
    if (isempty(rois))
        continue;
    end
    clf,imagesc2(I);
    [boxes,I] = get_mcg_boxes(conf,fra_db(t),roiParams);
    %     load(j2m('~/storage/s40_fra_selective_search',fra_db(t)));
    %     boxes = res.boxes;
    roiBoxes = cat(1,rois.bbox);
    [ovp,ints] = boxesOverlap(boxes,roiBoxes);
    ovp = max(ovp,[],2);
    plotBoxes(boxes(ovp>.5,:),'g-');
    pause;
    
end

%% show lips of drinking people, etc.

mouth_images = {};
all_images = {};
for t = 1:length(fra_db)
    if (mod(t,30)==0)
        t
    end
    if (~fra_db(t).isTrain),continue,end;
    %    if (fra_db(t).classID~=1),continue,end;
    [rois,~,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t));
    rois = rois([rois.id]==4);
    bbox = inflatebbox(rois.bbox,1.5,'both',false);
    mouth_images{t} = cropper(I,round(bbox));
    all_images{t} = I;
    %     clf; imagesc2(M);
    %     pause
end

%%


mImage(mouth_images);
mImage(all_images);
S = cellfun2(@(x) imResample(rgb2gray(x(end/4:3*end/4,end/4:3*end/4,:)),[80 80],'bilinear'),all_images);
mImage(S);

mImage(mouth_images(2:3:end));
multiWrite(mouth_images,'~/tmp_for_sal',{fra_db.imageID});
%%
sf_model = loadvar(fullfile(root_dir, 'datasets', 'models', 'sf_modelFinal.mat'),'model');
%%
addpath(genpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained'));
addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 50;
opts.pixNumInSP = spSize;
conf.get_full_image = true;

close all
doSelectiveSearch = true;
for iImage =5:1:500
    if (fra_db(iImage).classID~=3),continue,end
    roiParams1 = roiParams;
    roiParams1.infScale = 1.5;
    [rois,~,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(iImage),roiParams1);
    I = im2uint8(mouth_images{iImage});
    [candidates, ucm2] = im2mcg(I,'fast',true);
    [res,res_bd,resizeRatio] = extractSaliencyMap(imresize(I,2),opts);
    %    I = getImage(conf,fra_db(iImage));
    %    res = foregroundSaliency(conf,fra_db(iImage).imageID);
    clf; subplot(2,2,1); imagesc2(I);
    subplot(2,2,2); imagesc2(res);
    res = imResample(res,size2(I));
    Q = sc(cat(3,res,im2double(I)),'prob_jet');
    subplot(2,2,3); imagesc2(Q);
    
    %pause
    %
    % calculate mean intensity, area of each candidate.
    nCandidates  =length(candidates.scores);
    candidate_stats = zeros(nCandidates,5);
    for ii = 1:nCandidates
        m = res(candidates.masks(:,:,ii));
        candidate_stats(ii,1) = mean(m);
        candidate_stats(ii,2) = min(m);
        candidate_stats(ii,3) = median(m);
        candidate_stats(ii,4) = max(m);
        candidate_stats(ii,5) = nnz(m)^.5;
    end
    candidate_stats = [candidate_stats,candidates.scores];
    %
    %%
    scores = candidate_stats*[1 0 0 0 0 0]';
    [r,ir] = sort(scores,'descend');
    %     [r,ir] = sort(candidates.scores,'descend');
    
    %     % z = zeros(size2(I));
    %     % for t = 1:size(candidates.masks,3)
    %     %     z = z+double(candidates.masks(:,:,t)).*candidates.scores(t);
    %     % end
    %     %
    %     % figure,imagesc2(z)
    %         [r,ir] = sort(candidates.scores,'descend')
    close all
    
    clf
    for ik = 1:100%min(5,nCandidates)
        t = ir(ik)
        clf;displayRegions( Q, candidates.masks(:,:,t));
        pause(.1)
    end
    
    %     disp(iImage)
end

%%
load fra_db;
for iImage = 546:5:length(fra_db)
    
    roiParams.infScale = 1.5;
    roiParams.absScale = 200;
    if (fra_db(iImage).classID~=3),continue,end
    roiParams.centerOnMouth = false;
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(iImage),roiParams);
    
    clf;imagesc2(I);
    load(j2m('~/storage/fra_landmarks',fra_db(iImage).imageID));
    %landmarks = res.landmarks(1:6);
    landmarks = res.landmarks(2);
    
    %     [landmarks.rotation]
    
    all_scores = -inf(size(landmarks));
    for t = 1:length(landmarks)
        if (~isempty(landmarks(t).s))
            all_scores(t) = landmarks(t).s;
        end
    end
    
    %
    %     if (~isempty(t))
    clf;imagesc2(I);
    %     for t = 1:length(landmarks)
    %         if (~isempty(landmarks(t).s))
    %                     polys = cellfun2(@mean,landmarks(t).polys);
    %             clf; imagesc2(I); plotPolygons(polys,'g+');
    %             pause(.1)
    %         end
    %     end
    [u,t] = max(all_scores);
    if (~isinf(u))
        %if (~isempty(t))
        polys = cellfun2(@mean,landmarks(t).polys);
        plotPolygons(polys,'r*');
        
    end
    pause;
end
% scores = [res.landmarks.s];


%%
roiParams.infScale = 1.5;
roiParams.absScale = 200;
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);

%%cd '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;


cd '/home/amirro/code/3rdparty/mcg-2.0/full';install
[candidates, ucm2] = im2mcg(I,'fast',true);

%%
set_defaults;
set_custom_defaults;

% train exemplar svms using the circulant training, for each object class.


object_svms_path = '~/storage/misc/object_svms.mat';
if (exist(object_svms_path,'file'))
    load(object_svms_path);
else
    
    %object_svms = struct('classID',{},'sourceImageIndex',{},'detector',{});
    object_svms = struct;
    %     object_svms = {};
    %     train_classes = [fra_db(isTrain).classID];
    %training_data = struct('classID',{},'sourceImageIndex',{},'img',{},'obj_bbox',{});
    training_data = {};
    for t = 1:length(fra_db)
        t
        if (fra_db(t).isTrain)
            [rois,roiBox,I] = get_rois_fra(conf,fra_db(t),roiParams);
            f = find([rois.id]==3);
            if (isempty(f))
                continue;
            end
            f = f(1);
            rois = rois(f);
            scaleFactor = 90*curParams.extent/1.5 /size(I,1);
            orig_size = size2(I);
            I = imResample(I,scaleFactor);
            cur_data = struct;
            cur_data.classID = fra_db(t).classID;
            cur_data.sourceImageIndex = t;
            cur_data.img = I;
            cur_data.obj_bbox = rois.bbox*scaleFactor;
            training_data{end+1} = cur_data;
            %             clf; imagesc2(I); plotBoxes(cur_data.obj_bbox);
            %             pause
            %train_imgs{t} = I;
        end
    end
    
    training_data = cat(1,training_data{:});
    
    % train the svms
    imgClasses = [training_data.classID];
    allImgs = {training_data.img};
    %     for iClass = 1:4
    object_svms = training_data;
    object_svms(1).classifier = [];
    for u = 124:length(training_data)
        u
        curTrainingData = training_data(u);
        if (curTrainingData.classID==5),continue,end % don't want the phone class for now
        curClass = curTrainingData.classID;
        sel_neg = imgClasses~=iClass;
        bb = round(curTrainingData.obj_bbox);
        curImg = curTrainingData.img;
        bb = clip_to_image(bb,curImg);
        boxMask = poly2mask2(bb,size2(curImg));
        if (none(boxMask))
            disp(['warning - no object mask in image ' num2str(u)]);
            continue,
        end
        ff = fhog(im2single(curImg),8);
        boxMask_small = imResample(single(boxMask),size2(ff)) > 0;
        %         if (none(boxMask_small)) % if the mask disappeared, put a single pixel where it was centered.
        %             sizeRatio = size(ff,1)/size(curImg,1);
        %             bbCenter = boxCenters(bb);
        %             bbCenter = round(bbCenter*sizeRatio);
        %             boxMask_small(bbCenter(2),bbCenter(1)) = 1;
        %         end
        %             boxMask_small = imdilate(boxMask_small,ones(3)); % increase by 1 in each direction
        maskBox = region2Box(boxMask_small);
        %             maskBox = region2Box(imResample(single(boxMask),size2(ff))>0);
        ff = cropper(ff,maskBox);
        neg_imgs = allImgs(imgClasses~=curClass);
        neg_imgs = neg_imgs(1:3:end);
        %spause
        conf.features.winsize = size2(ff);
        curCluster = makeCluster(ff(:),[]);
        curCluster.winSize = size2(ff);
        classifier = train_circulant(conf,curCluster,neg_imgs);
        toShow = false;
        if (toShow)
            clf;
            V = hogDraw(ff,15,1);
            subplot(1,3,1); imagesc2(cropper(curImg,bb));
            subplot(1,3,2); imagesc2(V);
            V2 = hogDraw(reshape(classifier.w,curCluster.winSize(1),curCluster.winSize(2),[]),15,1);
            subplot(1,3,3);
            imagesc2(V2);
            pause
        end
        %struct('classID',{},'sourceImageIndex',{},'detector',{});
        %         object_svms(u) = training_data(u);
        object_svms(u).classifier = classifier;
        
    end
    save(object_svms_path,'object_svms');
    %     end
    %     curClass = train_classes(nn(t));
    %     pause
end

% do some testing...
for t = 1:length(fra_db)
    t
    if (fra_db(t).isTrain)
        [rois,roiBox,I] = get_rois_fra(conf,fra_db(t),roiParams);
        scaleFactor = 90*curParams.extent/1.5 /size(I,1);
        orig_size = size2(I);
        I = imResample(I,scaleFactor);
        conf.detection.params.detect_min_scale = .5;
        mask = true(size2(I));
        bias = 0;
        f_= struct('hog',1);
        close all
        allBoxes = {};
        
        
        object_svms(25).classifier = [];
        object_svms(28).classifier = [];
        
        I = imResample(I,1.5);
        
        for bb = 1:length(object_svms)
            bb
            curObjSVM = object_svms(bb);
            
            if (curObjSVM.classID==1 && ~isempty(curObjSVM.classifier))
                classifier = curObjSVM.classifier;
                boxes = my_quick_detect(conf,I,classifier.winSize,classifier.w);
                allBoxes{end+1} = boxes(1,:);
            end
        end
        
        allBoxes = cat(1,allBoxes{:});
        
        AA = computeHeatMap(I,allBoxes,'max');
        figure,imagesc2(AA)
        figure,imagesc2(I)
        
    end
end


%% face detection without bells and whistles...
initpath;
config;
visualizemodel(model)
%files = dir('~/storage/data/Stanford40/JPEGImages/*.jpg');
load fra_db;

roiParams.infScale = 1.5;
roiParams.absScale = -1;
imshow(I)

imgs = {};
for t  = 1:length(fra_db)
    t
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
    imgs{t} = I;
end

%%
% p =
cd ~/code/3rdparty/voc-release5/
startup
load ~/code/3rdparty/dpm_baseline.mat
randperm(length(files));
files = {fra_db.imageID};
faceDetections_baw = cell(size(fra_db));

%%
% z
for iFile = 1:1:length(imgs)
    %     iFile = p(iP);
    iFile
    %     if (isempty(faceDetections_baw{iFile}))
    %         I = imread(fullfile('~/storage/data/Stanford40/JPEGImages/',files{iFile}));
    %         I = imresize(I,2);
    clc;
    I = im2uint8(imgs{iFile});
    I = imresize(I,[128 NaN],'bilinear');
    [ds, bs] = imgdetect(I, model,-2);
    top = nms(ds, 0.1);
    ds = ds(top(1),:);
    
    %         ds(:,1:4) = ds(:,1:4);
    clf;
    %     imagesc2(I);
    showboxes(I, ds(:,:));
    ds(1,:)
    pause
    %         faceDetections_baw{iFile} = ds;
    %     end
end

% save faceDetections_baw.mat faceDetections_baw



%%
load fra_db;
initpath;
config;
%%
% z
for iFile = 201:length(fra_db)
    %     iFile = p(iP);
    iFile
    %     if (isempty(faceDetections_baw{iFile}))
    %         I = imread(fullfile('~/storage/data/Stanford40/JPEGImages/',files{iFile}));
    %         I = imresize(I,2);
    clc;
    R = j2m('~/storage/fra_faces_baw',fra_db(iFile));
    load(R);
    
    [rois,roiBox,I,scaleFactor] = get_rois_fra(conf,fra_db(iFile),res.roiParams);
    %         res.detections = res.detections(4);
    boxes = cat(1,res.detections.boxes);
    [u,iu] = sort(boxes(:,end),'descend');
    I = imrotate(I,res.detections(iu(1)).rot,'bilinear','crop');
    bb = round(boxes(iu(1),:));
    
    I = cropper(I,bb);
    clf;imagesc2(I); %plotBoxes(boxes(iu(1),:));
    boxes(iu(1),:)
    
    pause
    %         faceDetections_baw{iFile} = ds;
    %     end
end




%%
initpath;
config;

[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
%%
figure(1)
for id = length(train_ids):-1:1
    id
    conf.get_full_image = false;
    R = j2m('~/storage/s40_faces_baw',train_ids{id});
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    clf; imagesc2(getImage(conf,train_ids{id}));
    plotBoxes(detections.boxes(1,:));
    drawnow
    pause(.01)
end

%% faces on phrasal recognition dataset
bb = '~/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages/';
d = dir(fullfile(bb,'*'));

ress1 = {};
fileNames = {};
for u = 3:length(d)
    u
    R = j2m('~/storage/VOC3000_faces_baw',d(u).name);
    load(R);
    detections = res.detections;
    fileNames{end+1} = fullfile(bb,d(u).name);
    %     clf; imagesc2(imread(fullfile(bb,d(u).name)));
    detections = detections(3);
    if (isempty(detections.boxes))
        ress1{end+1} = zeros(1,5);
    else
        ress1{end+1} = detections.boxes(1,[1:4 6]);
    end
    %     plotBoxes(detections.boxes(1,:));
    %     drawnow
    %     pause(.1)
end
%%
figure(1)
rr = cat(1,ress1{:});
[r,ir] = sort(rr(:,end),'ascend');
for it = 2500:1:length(r)
    clf; imagesc2(imread(fileNames{ir(it)}));
    plotBoxes(rr(ir(it),:));
    drawnow
    pause(.1)
end

%%
L = load('allKPPreds.mat')
%% keypoint predictions
ppp = randperm(length(fra_db));
ppp = 1:length(fra_db);
%%
figure(1);clf;
mm = 2;
nn = mm;
for iClass =1:4
    n = 0;
    for it = 1:length(fra_db)
        it
        t = ppp(it);
        if (n>=mm*nn)
            break
        end
        
        if ~(fra_db(t).isTrain),continue,end
        roiParams.infScale = 1.5;
        roiParams.absScale = 192;
        roiParams.centerOnMouth = false;
        if (fra_db(t).classID==5),continue,end
        if (fra_db(t).classID~=iClass),continue,end
        [rois,roiBox,I] = get_rois_fra(conf,fra_db(t),roiParams);
        
        
        objInds = find([rois.id]==3);
        
        %         for t = 1:length(objInds)
        %         end
        %
        n = n+1;
        %     if (n==8)
        %         break,
        %     end
        vl_tightsubplot(mm,nn,n);
        %     clf;imagesc2(I);
        
        imagesc2(I);
        %         pause
        %         if  n > 3
        %                break
        %         end
        
        [preds,goods] = getKPPredictions(L,t);
        local_pred = squeeze(L.all_kp_predictions_local(t,:,:));
        global_pred = squeeze(L.all_kp_predictions_global(t,:,:));
        goods_1= global_pred(:,end) > 2;
        
        R = j2m('~/storage/fra_face_seg',fra_db(t));
        % % %         L1 = load(R);
        % % %         %     preds = (local_pred+global_pred)/2;
        % % %         candidates = L1.res.candidates;
        % % %         candidates.masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
        % % %         candidates.masks = squeeze(mat2cell2(candidates.masks,[1 1 size(candidates.masks,3)]));
        % % %         displayRegions(I,candidates.masks,[],0);
        preds = local_pred;
        bc1 = boxCenters(global_pred);
        bc2 = boxCenters(local_pred);
        bc_dist = sum((bc1-bc2).^2,2).^.5;
        bad_local = bc_dist > 30;
        %         preds(bad_local,1:4) = global_pred(bad_local,1:4);
        
        %         plotBoxes(preds(goods_1 & ~bad_local,:));
        plotBoxes(preds(9,:),'color','y','LineWidth',3);
        plotBoxes(preds(10,:),'color','r','LineWidth',3);
        plotBoxes(preds(11,:),'color','c','LineWidth',3);
        
        pointsOfInterest = preds(9,:);
        pointsOfInterest(any(isinf(pointsOfInterest),2),:) = [];
        roiSquare= makeSquare(pts2Box(pointsOfInterest),1);
        roiSquare = inflatebbox(roiSquare,80,'both',true);
        
        xlim(roiSquare([1 3]));
        ylim(roiSquare([2 4]));
        %
        %         mmm = 1;nnn = 2;qqq = 1;
        %         clf; subplot(mmm,nnn,qqq);
        %         II = cropper(I,round(roiSquare));
        %         imagesc2(II);
        %         qqq = qqq+1;
        %         subplot(mmm,nnn,qqq);
        %         %V = vl_slic(single(vl_xyz2lab(vl_rgb2xyz(im2single(II)))),20,.001);
        %         V = vl_slic(im2single(II),10,.01);
        %         imagesc2(paintSeg(II,V));
        %         pause
        % %         preds = boxCenters(preds);
        % %         plotPolygons(preds(goods_1 & ~bad_local,:),'g.','MarkerSize',3);
        % %         plotPolygons(preds(9,:),'y.','MarkerSize',3,'LineWidth',2);
        % %
        % %         plotPolygons(preds(10,:),'r.','MarkerSize',3,'LineWidth',2);
        % %         plotPolygons(preds(11,:),'c.','MarkerSize',3,'LineWidth',2);
        
        
        %     plotBoxes(preds(goods_1 & bad_local,:),'color','r','LineWidth',2);
        %             pause
        %     drawnow
        %     pause(.1)
    end
    pause
    %     saveas(gcf,fullfile('/home/amirro/notes/images/2014_09_18/faces',[classNames{iClass} '_facial_keypoints.png']));
end
%


%% show zhu aflw landmarks
clear all_lm_data
lmDir = '~/storage/aflw_zhu_landmarks';
d = dir(fullfile(lmDir,'*.mat'));
imgPaths = {};
for iImage = 1:length(d)
    iImage
    load(fullfile(lmDir,d(iImage).name));
    landmarks = [res.landmarks];
    for t = 1:length(landmarks)
        if (isempty(landmarks(t).s))
            landmarks(t).s = -inf;
        end
    end
    scores = [landmarks.s];
    [s,is] = max(scores);
    if (any(s))
        all_lm_data(iImage) = landmarks(is);
    else
        all_lm_data(iImage).s = -inf;
    end
    imgPaths{iImage} = fullfile('~/storage/data/aflw_cropped_context/',strrep(d(iImage).name,'.mat','.jpg'));
end


% save ~/storage/misc/zhu_aflw_landmarks.mat imgPaths all_lm_data

% for t = 1:length(all_lm_data)
%     if (isempty(all_lm_data(t).s))
%         'sdgf'
%         break
%     end
% end
allScores = [all_lm_data.s];
[s,is] = sort(allScores,'descend');
%%
close all
figure(1)
for k = 1:10:length(allScores)
    t = is(k);
    I = imread(imgPaths{t});
    clf; imagesc2(I);
    polys = all_lm_data(t).polys;
    polys = cellfun2(@(x) mean(x,1),polys);
    plotPolygons(polys,'r.');
    title(num2str(s(k)));
    drawnow
    pause
    %
    %     pause
end

%%
mm = 1;
nn = 3;
for k = 210:length(fra_db)
    k
    %     if (~train_labels(k)), continue, end;
    imageID = fra_db(k).imageID;
    out_dir = conf.elsdDir;
    %     imageID=  'blowing_bubbles_006.jpg';
    if (fra_db(k).classID~=2),continue,end
    resPath = fullfile(out_dir,strrep(imageID,'.jpg','.txt'));
    if (~exist(resPath,'file')), continue, end;
    %     L = load(resPath);
    [I,I_rect] = getImage(conf,imageID);
    A = dlmread(resPath);
    [lines_,ellipses_] = parse_svg(A,I_rect(1:2));
    
    clf,vl_tightsubplot(mm,nn,1);imagesc(1-I*0);axis image; hold on;
    
    %     figure,imagesc(I);axis image; hold on;
    plot_svg(lines_,ellipses_);
    vl_tightsubplot(mm,nn,2); imagesc(I); axis image; hold on;
    
    lineSegFile = fullfile(conf.lineSegDir,strrep(imageID,'.jpg','.mat'));
    L = load(lineSegFile);
    vl_tightsubplot(mm,nn,3)
    hold on; drawedgelist(L.edgelist,[],1,'rand');
    xlim(I_rect([1 3]));
    ylim(I_rect([2 4]));
    
    %     seglist = lineseg(L.edgelist,1);
    %           [ucm,gPb_thin] = loadUCM(conf,imageID);
    imageID
    pause
end

%% faces on afw...
bb = '~/data/afw/testimages';
d = dir(fullfile(bb,'*.jpg'));

ress1 = {};
fileNames = {};
inds = {};
t = 0;
for u = 1:length(d)
    u
    R = j2m('~/storage/afw_faces_baw',d(u).name);
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    
    %     clf; imagesc2(imread(fullfile(bb,d(u).name)));
    %     detections = detections(3);
    
    %         ress{end+1} = zeros(1,5);
    %         inds{end+1} = [];
    if (~isempty(detections.boxes))
        t = t+1;
        fileNames{end+1} = fullfile(bb,d(u).name);
        curBoxes= detections.boxes(1:min(1,size(detections.boxes,1)),[1:4 6]);
        ress1{end+1} = curBoxes;
        inds{end+1} = t*ones(size(ress1{end},1),1);
    end
    %     plotBoxes(detections.boxes(1,:));
    %     drawnow
    %     pause(.1)
end
%%

rr = cat(1,ress1{:});
inds = cat(1,inds{:});
%%
figure(1);
[r,ir] = sort(rr(:,end),'descend');
for it = 1:1:length(r)
    if (r(it)<.6),continue,end
    clf; imagesc2(imread(fileNames{inds(ir(it))}));
    %     rr(ir(it),:)
    plotBoxes(rr(ir(it),:));
    drawnow
    pause(.1)
end

%% convert normal face detections into fra_db structure for the full pipeline

%%
figure(1)
for id = 1:length(train_ids)
    id
    conf.get_full_image = false;
    R = j2m('~/storage/s40_faces_baw',train_ids{id});
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    clf; imagesc2(getImage(conf,train_ids{id}));
    plotBoxes(detections.boxes(1,:));
    drawnow
    pause(.01)
end
%%

conf.get_full_image = false;
s40_fra = fra_db(1);
Params.infScale = 1.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
roiParams.useCenterSquare = false;
roiParams.squareSide = 30*roiParams.absScale/105;
roiParams.centerOnMouth = false;

s40_fra = struct('imageID',{},'imageIndex',{},'isTrain',{},'faceBox',{});
for t = 1:length(train_ids)
    
    R = j2m('~/storage/s40_faces_baw',train_ids{t});
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    s40_fra(t).imageID = train_ids{t};
    s40_fra(t).faceBox = detections.boxes(1,1:4);
    % %     clf; imagesc2(getImage(conf,train_ids{t}));
    % %     plotBoxes(detections.boxes(1,1:4));
    [rois,roiBox,I] = get_rois_fra(conf,s40_fra(t),roiParams);
    
    %     I_orig = getImage(conf,train_ids{t});
    %     clf;imagesc2(I_orig); plotBoxes(roiBox);
    %     pause
end


%% create an fra_db for s40
load fra_db;
newImageData = augmentImageData(conf,[]);
fra_db_names = {fra_db.imageID};
all_names = {newImageData.imageID};
[aa,bb,cc] = intersect(fra_db_names,all_names);
imageIDS = {newImageData.imageID};
%%
%% aggregate all face detections
if (exist('~/storage/misc/s40_face_detections.mat','file'))
    load ~/storage/misc/s40_face_detections.mat;
else
    all_detections = struct('imageID',{},'detections',{});
    %
    for t = 1:length(imageIDS)
        if (mod(t,100)==0)
            t
        end
        R = j2m('~/storage/s40_faces_baw',imageIDS{t});
        load(R);
        all_detections(t).detections = res.detections;
        all_detections(t).imageID = imageIDS{t};
    end
    save ~/storage/misc/s40_face_detections.mat all_detections
end
%%
s40_fra = struct('imageID',{},'imgIndex',{},'isTrain',{},'faceBox',{},'indInFraDB',{},'objects',{},'mouth',{});
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
all_ids_orig = [train_ids;test_ids];
all_labels_orig = [all_train_labels;all_test_labels];
[aa1,bb1,cc1] = intersect(imageIDS,all_ids_orig);
load fra_db;
for t = 1:length(imageIDS)
    if (mod(t,100)==0)
        t
    end
    s40_fra(t).faceBox = -inf;
    s40_fra(t).valid = false;
    s40_fra(t).imgIndex = t;
    s40_fra(t).isTrain = newImageData(t).isTrain;
    detections = all_detections(t).detections;
    s40_fra(t).imageID = imageIDS{t};
    I_rect = newImageData(t).I_rect;
    s40_fra(t).raw_faceDetections = detections;
    s40_fra(t).faceBox = detections.boxes(1,1:4);
    s40_fra(t).indInFraDB = -1;
    if (all(isinf(s40_fra(t).faceBox)))
        continue;
    end
    s40_fra(t).valid = true;
    s40_fra(t).faceBox = s40_fra(t).faceBox+I_rect([1 2 1 2]);
    
    kp_preds = loadDetectedLandmarks(conf,s40_fra(t));
    s40_fra(t).mouth=boxCenters(kp_preds(3,:));
    %     I_orig = getImage(conf,s40_fra(t));
    %     clf; imagesc2(I_orig);
    %     plotBoxes(kp_preds(:,1:4),'g-');
    %     drawnow
    %     pause
end
for tt = 1:length(cc)
    s40_fra(cc(tt)).indInFraDB = bb(tt);
    cur_fra_db = fra_db(bb(tt));
    s40_fra(cc(tt)).faceBox_gt = cur_fra_db.faceBox;
    s40_fra(cc(tt)).mouth_gt = cur_fra_db.mouth;
    s40_fra(cc(tt)).objects_gt = cur_fra_db.objects;
    %     %     s40_fra(bb(u)) = cur_s40_fra;
end

[aa1,bb1,cc1] = intersect(imageIDS,all_ids_orig);
for u = 1:length(bb1)
    s40_fra(bb1(u)).classID = all_labels_orig(cc1(u));
end

save s40_fra s40_fra
%%

inds_in_fra = [s40_fra.indInFraDB];
D = defaultPipelineParams(false);
D.debug = true;
conf.get_full_image = true;
for t = 1:length(inds_in_fra)
    if (inds_in_fra(t)==-1),continue,end
    %     break
    curImgData = s40_fra(t);
    if (curImgData.isTrain),continue,end
    extract_all_features(conf,curImgData,D);
end

%%
% load fra_db;
dParams = defaultPipelineParams(false);
dParams.debug = true;
close all
t = 955;
curFeats = extract_all_features(conf,s40_fra(t),dParams);
[I_orig,I_rect] = getImage(conf,s40_fra(t));
x2(I_orig); plotBoxes(I_rect); plotBoxes(s40_fra(t).faceBox);x2(I_orig);
plotBoxes(bsxfun(@plus,s40_fra(t).raw_faceDetections.boxes,[I_rect([1 2 1 2]) 0 0]));



%%
%%
initpath;
config;
load s40_fra;
%%
nImages = length(s40_fra);
top_face_scores = zeros(nImages,1);
for t = 1:nImages
    top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
end
min_face_score = 0;
validImages = top_face_scores > 0;

% check on train set...
% nnz(top_face_scores>0)
sel_ = ~[s40_fra.isTrain] & [s40_fra.classID]==conf.class_enum.DRINKING;
% figure,plot(top_face_scores(sel_))
nnz(top_face_scores(sel_)>0)/nnz(sel_)
nnz(top_face_scores>0)/nImages;
all_dnn_feats = zeros(4096,nImages);
validImages = top_face_scores > 0;
net = init_nn_network();
Is = {};
conf.get_full_image = true;
sel_imgs = find(top_face_scores > min_face_score);
batchSize = 256;
% for t = 1:batchSize:length(sel_imgs)
%     img_ind = sel_imgs(t)
%     Is{t} = im2uint8(getImage(conf,s40_fra(img_ind).imageID));
x_global = extractDNNFeatsHelper(conf,{s40_fra(sel_imgs).imageID},net);
% x_global = extractDNNFeatsHelper(conf,s40_fra(sel_imgs),net);

%     imo = prepareForDNN(Is,net);
% end

%% get the global and local feats for all fra_db images...
outDir = '~/storage/s40_fra_global_dnn_feats_fc7';
clear all_s40_dnn;
for t = 1:nImages
    t
    R = j2m(outDir,s40_fra(t));
    all_s40_dnn(t) = load(R);
end

save ~/storage/misc/all_s40_dnn_fc7 all_s40_dnn;


%% 128
outDir = '~/storage/s40_fra_global_dnn_feats_m_128';
clear all_s40_dnn_128;
for t = 1:nImages
    t
    R = j2m(outDir,s40_fra(t));
    all_s40_dnn_128(t) = load(R);
end

save ~/storage/misc/all_s40_dnn_128 all_s40_dnn_128;
aaa = [all_s40_dnn_128.crop_feat];
%% 2048
outDir = '~/storage/s40_fra_global_dnn_feats_m_2048';
clear all_s40_dnn_m_2048;
for t = 1:nImages
    t
    R = j2m(outDir,s40_fra(t));
    all_s40_dnn_m_2048(t) = load(R);
end

save ~/storage/misc/all_s40_dnn_m_2028 all_s40_dnn_m_2048;

%% very deep

% aaa = [all_s40_dnn_128.crop_feat];
outDir = '~/storage/s40_fra_global_dnn_feats_verydeep';
clear all_s40_dnn_verydeep;
for t = 1:nImages
    t
    R = j2m(outDir,s40_fra(t));
    all_s40_dnn_verydeep(t) = load(R);
end

save ~/storage/misc/all_s40_dnn_verydeep all_s40_dnn_verydeep;

%%
train_imgs = [s40_fra.isTrain];

sel_class = [s40_fra.classID]==conf.class_enum.DRINKING;
all_feats = [all_s40_dnn.crop_feat];
test_labels = sel_class(~train_imgs);
[posFeats,negFeats] = splitFeats(all_feats(:,train_imgs),sel_class(train_imgs));
classifier = train_classifier_pegasos(posFeats,negFeats,0);
%%
test_res = classifier.w(1:end-1)'*all_feats(:,~train_imgs);
%%
clf;figure(1)
test_rest(~validImages(~train_imgs)) = -inf;
vl_pr(test_labels*2-1,test_res(:));
%%
[r,ir] = sort(test_res,'descend');
imgs = s40_fra(~train_imgs);
displayImageSeries(conf,{imgs(ir).imageID});

%% make a new version of s40_fra, this time each one contains multiple faces.
%%

s40_fra = struct('imageID',{},'imgIndex',{},'isTrain',{},...
    'detected_faces',{});

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
all_ids_orig = [train_ids;test_ids];
all_labels_orig = [all_train_labels;all_test_labels];
[aa1,bb1,cc1] = intersect(imageIDS,all_ids_orig);
load fra_db;

for t = 1:length(imageIDS)
    if (mod(t,100)==0)
        t
    end
    s40_fra(t).faceBox = -inf;
    s40_fra(t).valid = false;
    s40_fra(t).imgIndex = t;
    s40_fra(t).isTrain = newImageData(t).isTrain;
    s40_fra(t).imageID = imageIDS{t};
    s40_fra(t).indInFraDB = -1;
    s40_fra(t).valid = false;
    detections = all_detections(t).detections;
    I_rect = newImageData(t).I_rect;
    s40_fra(t).I_rect = I_rect;
    %s40_fra(t).raw_faceDetections = detections;
    clear detected_faces;
    %%'faceBox',{},'indInFraDB',{},'objects',{},'mouth',{});
    if (all(isinf(s40_fra(t).detections.boxes)))
        continue;
    end
    s40_fra(t).valid = true;
    for iDet = 1:size(detections.boxes,1)
        curBox = detections.boxes(iDet,:);
        curBox(1:4) = curBox(1:4) + I_rect([1 2 1 2]);
        detected_faces(iDet).faceBox = curBox;
        detected_faces(iDet).faceScore = curBox(6);
    end
    s40_fra(t).detected_faces = detected_faces;
end
for tt = 1:length(cc)
    s40_fra(cc(tt)).indInFraDB = bb(tt);
    cur_fra_db = fra_db(bb(tt));
    s40_fra(cc(tt)).faceBox_gt = cur_fra_db.faceBox;
    s40_fra(cc(tt)).mouth_gt = cur_fra_db.mouth;
    s40_fra(cc(tt)).objects_gt = cur_fra_db.objects;
    %     %     s40_fra(bb(u)) = cur_s40_fra;
end

[aa1,bb1,cc1] = intersect(imageIDS,all_ids_orig);
for u = 1:length(bb1)
    s40_fra(bb1(u)).classID = all_labels_orig(cc1(u));
end


%% run face detection on COFW
R = load('/home/amirro/code/3rdparty/rcpr_v1/data/COFW_test.mat');
cd /home/amirro/code/3rdparty/voc-release5
startup
load ~/code/3rdparty/dpm_baseline.mat
addpath('~/code/utils');
clear res;
%%
% % for t = 1:length(R.IsT)
% %     t/length(R.IsT)
% %     I_orig = IsT{t};
% %     detections = struct('rot',{},'boxes',{});
% %     rots = 0;
% %     for iRot = 1:length(rots)
% %         I = imrotate(I_orig,rots(iRot),'bilinear','crop');
% %         [ds, bs] = imgdetect(I, model,-2);
% %         top = nms(ds, 0.1);
% %         if (isempty(top))
% %             boxes = -inf(1,5);
% %         end
% %         detections(iRot).rot = rots(iRot);
% %         detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
% %     end
% %     res(t).detections = detections;
% % %      figure(1); clf; imagesc2(I);plotBoxes(detections(1).boxes(1,:))
% % end
% %
% % save cofw_dets res


%% upper body...
for t = 1:length(s40_fra)
    if (s40_fra(t).classID~=conf.class_enum.FIXING_A_BIKE),continue,end
    conf.get_full_image = true;
    [I,I_rect]=  getImage(conf,s40_fra(t));
    
    R = load(j2m('/home/amirro/storage/s40_upper_body_2',s40_fra(t)));
    
    clf;imagesc2(I); plotBoxes(R.boxes)
    faceDetBoxes = s40_fra(t).raw_faceDetections.boxes;
    faceDetBoxes = faceDetBoxes(:,1:4)+repmat(I_rect([1 2 1 2]),size(faceDetBoxes,1),1);
    plotBoxes(faceDetBoxes,'b-','LineWidth',2);
    plotBoxes(faceDetBoxes(1,:),'r--','LineWidth',2);
    plotBoxes(R.boxes(1:min(size(R.boxes,1),2),:),'m--','LineWidth',2);
    pause
end

%% test the new annotations
faceDetectionDir = '~/storage/faces_only_baw';
load ~/code/mircs/s40_fra_faces.mat;

load s40_fra.mat;
%%
s40_fra_faces_d = s40_fra_faces;
for t = 1:1:length(s40_fra_faces)
    t
    curImgData =s40_fra_faces(t);
    if (s40_fra(t).isTrain && s40_fra(t).indInFraDB~=-1)
        continue;
    end
    t
    %     break
    
    
    %     [I,I_rect] = getImage(conf,curImgData);
    %     clf; imagesc2(I); plotBoxes(curImgData.faceBox);
    load(j2m(faceDetectionDir,curImgData));
    if (~isinf(detections.boxes(1)))
        s40_fra_faces_d(t).faceBox = detections.boxes(1,1:4);
    else
        error('inf!!!!')
    end
    %
    %     plotBoxes(detections.boxes(1,:),'r--','LineWidth',2);
    %     pause
end
s40_fra_faces_d(6217).faceBox = [101 50 124 74];
s40_fra_faces_d(1540).faceBox = [298 47 318 71];
s40_fra_faces_d(7927).faceBox = [208 84 219 96];
s40_fra_faces_d(8842).faceBox = [406 116 428 140];
s40_fra_faces_d(8953).faceBox = [260 86 276 112];

save ~/code/mircs/s40_fra_faces_d s40_fra_faces_d;
% for t = 1:length(s40_fra_faces_d)
%     s40_fra_faces_d(t).valid=true
% end

%% add mouths....
load s40_fra_faces_d;
% s40_fra_faces_d = s40_fra_faces_d;
% faceDetectionDir = '~/storage/faces_only_baw';
kpPredDir = '~/storage/faces_only_landmarks';
for t = 1:1:length(s40_fra_faces_d)
    t
    
    s40_fra_faces_d(t).valid = true;
    if (s40_fra_faces_d(t).isTrain)
        continue
    end
    if (s40_fra_faces_d(t).isTrain && s40_fra_faces_d(t).indInFraDB~=-1)
        s40_fra_faces_d(t).valid = true;
        continue
    end
    imageID = s40_fra_faces_d(t).imageID;
    
    curOutPath = j2m(kpPredDir,imageID);
    L = load(curOutPath);%,'curKP_global','curKP_local');
    if (~isfield(L,'res'))
        res = L;
    end
    
    
    global_pred = res.kp_global;
    local_pred = res.kp_local;
    preds = local_pred;
    bc1 = boxCenters(global_pred);
    bc2 = boxCenters(local_pred);
    bc_dist = sum((bc1-bc2).^2,2).^.5;
    bad_local = bc_dist > 30;
    goods_1= global_pred(:,end) > 2;
    local_pred(bad_local,1:4) = global_pred(bad_local,1:4);
    goods = goods_1 & ~bad_local;
    kp_preds = local_pred;
    goods = true(size(kp_preds,1),1);
    kp_centers = boxCenters(kp_preds);
    nKP = size(kp_centers,1);
    
    %     clf;imagesc2(getImage(conf,imageID))
    %     plotBoxes(s40_fra_faces_d(t).faceBox)
    %     plotPolygons(kp_centers,'g.')
    s40_fra_faces_d(t).mouth = boxCenters(kp_preds(3,:));%/scaleFactor+roiBox(1:2);
    %     plotPolygons(mouth,'r*');
    %     drawnow
    %     pause
    continue
    
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    % [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_struct,roiParams);
    
    %fra_struct.mouth = boxCenters(kp_preds(3,:))/scaleFactor+roiBox(1:2);
    
    %     s40_fra_faces_d(t).valid = s.valid;
    %     if (s.valid)
    %         s40_fra_faces_d(t).mouth = s.mouth;
    %     else
    %         '!!!!!'
    %         break
    %     end
end

save ~/code/mircs/s40_fra_faces_d s40_fra_faces_d;
%%
faceDetectionDir = '~/storage/faces_only_baw_2';
% get face scores...
for t = 1760:length(s40_fra_faces_d)
    t
    s40_fra_faces_d(t).faceScore = -inf;
    if (s40_fra_faces_d(t).isTrain && s40_fra_faces_d(t).indInFraDB~=-1)
        s40_fra_faces_d(t).valid = true;
        continue
    end
    imageID = s40_fra_faces_d(t).imageID;
    
    R = j2m(faceDetectionDir,imageID);
    %R = j2m('~/storage/s40_faces_baw',imageID);
    
    %     if (~exist(R,'file'))
    %         error('face detection file doesn''t exist');
    %     end
    r = load(R);
    detections = r.detections;
    s40_fra_faces_d(t).faceScore = detections.boxes(1,end);
end

save ~/code/mircs/s40_fra_faces_d s40_fra_faces_d;

%%
% load s40_fra_faces_d;
% % s40_fra_faces_d = s40_fra_faces_d;
% faceDetectionDir = '~/storage/faces_only_baw';
% for t = 1:length(s40_fra_faces_d)
%     t
%     U = j2m('~/storage/face_only_feature_pipeline_all',s40_fra_faces_d(t));
%     if (exist(U,'file'))
%         L = load(U,'feats','moreData');
%         save(U,'-struct','L');
%     end
% end
%
%
% d =dir('~/storage/face_only_feature_pipeline_all');
% n = 0;
% bytes = [d.bytes];
% for u = 1:length(d)
%
%     if any(strfind(d(u).date,'11-Nov-2014 01:'))
%        if (d(u).bytes)/10^6 < 10
%            delete(fullfile('~/storage/face_only_feature_pipeline_all',d(u).name))
%            u
%        end
%     end
% end

s = face_detection_to_fra_struct(conf,faceDetectionDir,'cleaning_the_floor_010.jpg');

dirsToClean = {'~/storage/faces_only_baw','~/storage/faces_only_landmarks','~/storage/faces_only_seg',...
    '~/storage/s40_obj_prediction_faces_only','~/storage/face_only_feature_pipeline_all'};

for t = 1:length(s40_fra_faces_d)
    if (~s40_fra_faces_d(t).valid)
        s40_fra_faces_d(t)
        t
        for ii = 1:length(dirsToClean)
            delete(j2m(dirsToClean{ii},s40_fra_faces_d(t)));
        end
    end
end


%% get paths for many non-person images
ids = getNonPersonIds(VOCopts);
non_person_paths = {}
for t = 1:length(ids)
    non_person_paths{t} = sprintf(VOCopts.imgpath,ids{t});
end

save ~/storage/misc/non_person_paths non_person_paths


%%
% load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
load ~/storage/misc/s40_fra_faces_d_new
fra_db = s40_fra_faces_d
%%
conf.get_full_image = true;
roiParams = defaultROIParams();
ptNames = landmarkParams.ptsData;
ptNames = {ptNames.pointNames};
requiredKeypoints = unique(cat(1,ptNames{:}));
landmarkParams = load('~/storage/misc/kp_pred_data.mat');
landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX,'Distance','L2');
landmarkParams.conf = conf;
landmarkParams.wSize = 96;
landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
landmarkParams.requiredKeypoints = requiredKeypoints
%{'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};


landmarkInit = landmarkParams;
landmarkInit.debug_ = false;
%%

%% 1. get zhu keypoints for aflw
%% initialize two models: profile and frontal
%% initialize rcpr using nearest neighbors....
figure(1)
all_kp_global = zeros(length(requiredKeypoints),5,length(fra_db));
load ~/storage/misc/s40_face_detections.mat; % all_detections
fra_db = s40_fra_faces_d;
all_kp_global_orig_coordinates = zeros(size(all_kp_global));

%%
for u = 1:length(fra_db)
    u
    imgData = fra_db(u);
    if ~imgData.valid
        continue
    end
    %[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,imgData,roiParams);
    %faceBox = rois(1).bbox;
    %bb = round(faceBox(1,1:4));
    curFaceBox = all_detections(u).detections.boxes(1,:);
    curScore = curFaceBox(end);
    %if (curScore<.2),continue,end
    curFaceBox = curFaceBox(1:4);
    curFaceBox= round(inflatebbox(curFaceBox,1,'both',false));
    conf.get_full_image = false;
    [I_orig,I_rect] = getImage(conf,imgData);
    I = cropper(I_orig,curFaceBox);
    I_crop_orig = I;
    resizeFactor = 128/size(I,1);
    I = imResample(I,[128 128],'bilinear');
    bb = [1 1 fliplr(size2(I))];
    [kp_global] = myFindFacialKeyPoints(conf,I,bb,landmarkInit.XX,...
        landmarkInit.kdtree,landmarkInit.curImgs,landmarkInit.ress,landmarkInit.ptsData,landmarkInit);
    %     clf; imagesc2(I); plotBoxes(kp_global); pause;continue
    %     all_kp_global(:,:,u) = kp_global;
    
    %     kp_global = squeeze(all_kp_global(:,:,u));
    kp_global(:,1:4) = kp_global(:,1:4)/resizeFactor;
    
    kp_global(:,1:4) = bsxfun(@plus,kp_global(:,1:4),I_rect([1 2 1 2])+curFaceBox([1 2 1 2])-1);
    all_kp_global_orig_coordinates(:,:,u) = kp_global;
    %
    %     conf.get_full_image = true;
    %     [I_orig,I_rect] = getImage(conf,imgData);
    %     clf;imagesc2(I_orig);
    %     plotBoxes(I_rect([1 2 1 2])+curFaceBox);
    %     plotBoxes(kp_global);pause
    
    %
end
save all_kp_global_orig_coordinates all_kp_global_orig_coordinates


%%  casia webfaces
baseDir = '~/storage/CASIA_faces/CASIA-WebFace/';
d = dir(baseDir);
%%
images = {}
for u = 3:5:length(d)
    u/length(d)
    images{end+1} = multiRead(conf,fullfile(baseDir,d(u).name),'jpg',[],[64 64]);
end

images = cat(2,images{:});

x2(images(1:100:end));


%%
%%load ~/storage/misc/s40_fra_faces_d_new

figure(1);
imagesDir = '/home/amirro/storage/data/Stanford40/JPEGImages';
% for t = 1:length(fra_db)
%     imgData = fra_db(t);
%     imagePath = fullfile(imagesDir,imgData(t).imageID);
%     I = imread(imagePath);
%     objects_mask = poly2mask2
%     displayRegions(I,imgData.objects
%

%%
%%
initpath
config;
load fra_db;
%addpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained');
addpath('/home/amirro/code/3rdparty/mcg-2.0/full');install

% install
%%
inDir = '~/storage/fra_db_seg';
outDir = '~/storage/fra_db_seg_simplified';
ensuredir(outDir);
for t = 16:length(fra_db)
    t
    imgData = fra_db(t);
    load(j2m(inDir,imgData));candidates = cadidates; clear cadidates;
    
    %     I = im2uint8(getImage(conf,imgData));
    
    %     I = imResample(I,.25,'bilinear');
    
    %     x2(I);
    %     res.cadidates = candidates
    %     res.ucm2 = ucm2
    %     [candidates,ucm2] = im2mcg(I,'accurate',false);
    
    masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
    masks = row(squeeze(mat2cell2(masks,[1,1,size(masks,3)])));
    polys = compressMasks(masks);
    candidates.bboxes = candidates.bboxes(:,[2 1 4 3]);
    
    for tt = 1:length(polys)
        %         t
        polys{tt} = uint16(polys{tt});
    end
    %     for t = 1:length(polys)
    %         clf;
    %         imagesc2(I);
    %         plotPolygons(polys{t},'r-','LineWidth',2);
    %         plotBoxes(candidates.bboxes(t,:));
    %         dpc
    %     end
    %
    %     polys = {};
    %     for ii = 1:length(masks)
    %         M = masks{ii};
    %         B = fliplr(bwtraceboundary2(M));
    % %         x2(poly2mask2(B,size2(M))-M)
    % %         simplifyPolygon
    %         B1 = simplifyPolygon(B,1);
    %         polys{ii} = B1;
    %         continue;
    %
    %         %continue
    %         clf; subplot(1,2,1);
    %         imagesc2(M);
    %         plotPolygons(B,'r-','LineWidth',3);
    %         size(B1,1)/size(B,1)
    %         plotPolygons(B1,'g-+');
    %         subplot(1,2,2); imagesc2(xor(poly2mask2(B,size2(M)),poly2mask2(B1,size2(M))));
    %         dpc
    % %         if ii > 5
    % %             break
    % %         end
    %     end
    
    
    res = struct;
    res.candidates.bboxes = candidates.bboxes;
    res.candidates.polys = polys;
    res.candidates.scores = candidates.scores;
    res.ucm2 = ucm2;
    
    save(j2m(outDir,imgData),'-struct','res');
end

%%
z = {};
for t = 1:35
    z{t} = single(rand(50));
end

max(col(abs(cellfun3(@(x) imResample(x,.5),z,3)-imResample(cat(3,z{:}),.5))))

%%
fid = fopen('~/code/mircs/out.txt');
fid2= fopen('dlib_face_landmarks.txt');
% file names...
s = fgetl(fid);
filePaths = {}
while s~=-1
    %         if (t >=1000),break,end
    filePaths{end+1} = s;
    f = fgetl(fid);
    nFaces = str2num(f);
    curLM = {};
    for u = 1:length(nFaces) % skip face lines
        f = fgetl(fid);
        LM = fgetl(fid2);
        LM = strrep(LM,'(','');
        LM = strrep(LM,')','');
        LM = strrep(LM,',','');
%        curLM{end+1} =
    end
    
    s = fgetl(fid);
end
fclose(fid)
fclose(fid2)

%%
%L = load('~/storage/misc/s40_tiny.mat');
load ~/storage/mircs_18_11_2014/s40_fra;
fra_db = s40_fra;
f = find([fra_db.classID] == conf.class_enum.BRUSHING_TEETH);
%x2(L.s40_tiny_people(f))

%%
ticID = ticStatus('shrinking s40',.5,.5,true);
S = {};
for it = 1:length(f)%1:length(s40_fra)
    %     t
    t = f(it);
    [I,I_rect] = getImage(conf,s40_fra(t));
    %s40_tiny{t} = imResample(I,[32 32],'bilinear',true);
    faceBox = s40_fra(t).raw_faceDetections.boxes(1,1:4);
    faceBox = faceBox+I_rect([1 2 1 2]);
    faceBox = round(inflatebbox(faceBox,1.2,'both',false));
    I_person = cropper(I,faceBox);
    resizeFactor = 64/size(I_person,1);
    %s40_tiny_people{t} = imResample(I_person,resizeFactor,'bilinear',true);
    S{end+1} = imResample(I_person,resizeFactor,'bilinear',true);
    tocStatus(ticID,it/length(f))
end
% tocStatus(ticID,1);
% save ~/storage/misc/s40_tiny.mat s40_tiny s40_tiny_people
%%
% L = load('~/temp.mat');
% a = L.a;
% a = bsxfun(@rdivide, a(:,1:3),a(:,4))
% ddd = 1:10:size(a,1);
% plot3(a(ddd,1),a(ddd,2),a(ddd,3),'r.')
%x2(S)%%
for u = 1:length(S)
    I = S{u};
    I = imResample(I,[18 18],'bilinear');
    im = im2single(I);
    im = vl_rgb2xyz(im);
    im = vl_xyz2lab(im);
    segs = vl_slic(single(im), 5, 1);
    [segImage,c] = paintSeg(I,segs);
    nSegs = max(segs(:));
    
    disp(['negs: ',num2str(nSegs)]);
    mm = 1;
    nn = 3;
    clf;
    
    subplot(mm,nn,3)
    imagesc2(segImage);
    
    subplot(mm,nn,1);
    imagesc2(I);
    
    subplot(mm,nn,2);
    z = {};
    for t = 1:nSegs
        z{t} = segs==t;
    end
    %displayRegions(I,z,[],0,3);
    displayRegions(I,z)
    %imagesc2(segImage);
    
    clc
    %     dpc
end


%%
curSet = f_train_pos;
for it = 1:1:length(curSet)
    t = curSet(it);
    t
    imgData = fra_db(t);
    I =  getImage(conf,imgData);
    faceBox = imgData.faceBox;
    mouthCenter = imgData.mouth;
    h = faceBox(3)-faceBox(1);
    mouthBox = round(inflatebbox(mouthCenter,[h/2 h/2],'both',true));
    I_sub = cropper(I,mouthBox);
    mouthMask = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib);
    %x2(I_sub);
    
%     I = imResample(I_sub,[32 32],'bilinear');
    %     #[cands,ucm2] = im2mcg(I,'accurate',true);
    I = I_sub;
    
    [Iseg labels map gaps E] = vl_quickseg(I, .5, 3,5);
    
    im = im2single(I);
    im = vl_rgb2xyz(im);
    im = vl_xyz2lab(im);
    segs = vl_slic(single(im), 10, 1);
    [segImage,c] = paintSeg(I,segs);
    nSegs = max(segs(:));
    
    disp(['negs: ',num2str(nSegs)]);
    mm = 2;
    nn = 2;
    clf;
    
    subplot(mm,nn,2)
    imagesc2(segImage);
    
    subplot(mm,nn,1);
    imagesc2(I);

    
    subplot(mm,nn,3);imagesc2(Iseg);
    dpc
    
    % %     subplot(mm,nn,2);
    % %     z = {};
    % %     for t = 1:nSegs
    % %         z{t} = segs==t;
    % %     end
    % %     %displayRegions(I,z,[],0,3);
    % %     displayRegions(I,z)
    %dpc
end

%% 


load fra_db;
images = {};
labels = {};
isTrain = [fra_db.isTrain];
dlib_landmark_split;
roiParams.infScale = 1;
for t = 1:length(fra_db)
    t
%     if ~fra_db(t).isTrain,continue,end
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
    
    if ~strcmp(rois(3).name,'obj'),
        continue;
    end
    
%     clf; 
%     imagesc2(I);
%     plotPolygons(rois(3).poly,'r-','LineWidth',2);
    [ bw ] = poly2mask2(rois(3).poly,size2(I));     
    images{end+1} = I;
    labels{end+1} = bw;   
%     
%     [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
%     [mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
%     [candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub);
%     candidates.masks = processRegions(I_sub,candidates,mouthMask); % remove some bad regions
%     [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
end

save images_and_labels images labels


%% correct all failed segmentations


for t = 1:length(s40_fra)
    imgData = s40_fra(t);
    segPath = j2m('~/storage/fra_db_mouth_seg_2',imgData);
    load(segPath);
    % correct train, if needed
    if any(~[segs.success])
        delete(segPath);
    end
end

%%
conf.get_full_image=true;
% get the vgg face...
modelPath = '/home/amirro/storage/matconv_data/vgg-face.mat';
net = load(opts.modelPath);
vl_setupnn
% net = dagnn.DagNN.fromSimpleNN(net.net, 'canonicalNames', true) ;
% net = dagnn.DagNN.loadobj(net.net);
net = load(modelPath) ;
net.layers = net.layers(1:33);
net = vl_simplenn_move(net,'gpu');

im = imread('https://upload.wikimedia.org/wikipedia/commons/4/4a/Aamir_Khan_March_2015.jpg') ;
% I1 = im;
%%
im = I1;
im = im(1:250,:,:) ; % crop
% im = imcrop(im);
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.normalization.averageImage);
%%
tic; 
res = vl_simplenn(net, gpuArray(im_));
toc
%%
desc = res34;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
              net.classes.description{best}, best, bestScore), ...
      'Interpreter', 'none') ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ; axis equal off ;
title(sprintf('%s (%d), score %.3f',...
              net.classes.description{best}, best, bestScore), ...
      'Interpreter', 'none') ;

%%
% get the images

load ~/storage/misc/images_and_masks_x2_w_hands.mat

train= find(isTrain);
test = find(~isTrain);
val = train(1:3:end);
train = setdiff(train,val);

imdb.images = 1:length(images);
imdb.images_data = images;
needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
if needToConvert
    for t = 1:length(images)
        imdb.images_data{t} = im2uint8(imdb.images_data{t});
    end
end
imdb.labels = masks;
imdb.nClasses = 3;

for t = 1:length(images)
    clf; imagesc2(images{t});
    plotPolygons(landmarks{t}(:,1:2),'g+','LineWidth',2);
    showCoords(landmarks{t}(:,1:2))
    dpc
end

% get descriptors for all.
descs = {};
for t = 1:length(fra_db)
    if (mod(t,20)==0),
        t
    end
    I1 = images{t};
    im_ = single(I1) ; % note: 255 range
    im_ = imResample(im_, net.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.normalization.averageImage);    
    res = vl_simplenn(net, gpuArray(im_));
    descs{t} = gather(res(end).x(:));    
end

all_descs = cat(2,descs{:});

%%
sel_train = isTrain;
%all_kps = reshape(land
all_kps = zeros(length(images),7,2);
for t = 1:length(landmarks)
    curLandmarks = landmarks{t}(:,1:2);
    bads = any(curLandmarks<=0,2);
    curLandmarks(bads,:) = nan;
    all_kps(t,:,:) = curLandmarks;
end

for t = 1:length(landmarks)    
    all_kps(t,:,:) = all_kps(t,:,:)/size(images{t},1);
end

kpNames = {'eye_left','eye_right','mouth_center','mouth_left','mouth_right','chin','nose'};
lambda = .1;
clear train;
%addpath('/home/amirro/code/3rdparty/libsvm-3.17/matlab');
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

%%
all_descs = cat(2,descs{:});
all_descs = normalize_vec(all_descs);
%%

lambda = .0001;
predictors = train_predictors(all_descs,find(sel_train),all_kps,kpNames,lambda)
%

preds_xy = apply_predictors(predictors,all_descs,find((~isTrain)));

%
for t = 1:length(fra_db)
    t
    imgData = fra_db(t);
    if imgData.isTrain,continue,end
    clf; imagesc2(images{t});
%     break;
    curPrediction = squeeze(preds_xy(t,:,:));
    curPrediction = curPrediction*size(images{t},1);
    plotPolygons(curPrediction,'r+','LineWidth',2);
    dpc
%     faceBox = inflatebbox(imgData.faceBox,1.2,'both',false);
%     I1 = cropper(I,round(faceBox));    
%     images{t} = I1;    
end


% 
%% 

images = {};
descs = {};
for t = 1:length(fra_db)
    t
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    faceBox = inflatebbox(imgData.faceBox,1.2,'both',false);
    I1 = cropper(I,round(faceBox));    
    images{t} = I1;    
end

descs = {};
for t = 1:length(fra_db)
    if (mod(t,20)==0),
        t
    end
    I1 = images{t};
    im_ = single(I1) ; % note: 255 range
    im_ = imResample(im_, net.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.normalization.averageImage);    
    res = vl_simplenn(net, gpuArray(im_));
    descs{t} = gather(res(end).x(:));    
end

 
%%
addpath(genpath('~/code/utils'));
D = normalize_vec(cat(2,descs{:}));
showNN(D,images,15,false);
