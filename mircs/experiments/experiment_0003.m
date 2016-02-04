%%%%%% Experiment 3 %%%%%%%
% Oct. 27, 2013

% re-use the old idea of distinguishing between the facial actions, by
% training a classifier. 

% 1. 

function experiment_0003(conf)

% initpath;
% config;
conf.class_subset = conf.class_enum.DRINKING;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
outDir = '~/mircs/experiments/experiment_0003';
mkdir(outDir);
conf.max_image_size = inf;
% prepare the data...
load ~/storage/misc/imageData_new;
load ~/mircs/experiments/common/faces_cropped_new.mat;

%%

multiWrite(faces.train_faces(imageData.train.labels),fullfile(outDir,'faces'));

%subDirs = dir(fullfile(outDir,'faces'));
subDirs = {'drink_side','cup_front','cup_side'};
pClassifiers = makeCluster(zeros(1984,1),[]);
iSubDir = 1;
for iSubDir = 1:length(subDirs)
    curSubDir = fullfile(outDir,'faces',subDirs{iSubDir});    
    %imgs = dir(fullfile(curSubDir,'*.jpg'));    
    flippedSubDir = fullfile(curSubDir,'flipped');
    m = multiRead(conf,flippedSubDir);      
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [10 10];
    X = imageSetFeatures2(conf,m,true,conf.features.winsize*conf.features.vlfeat.cellsize);
    conf.clustering.num_hard_mining_iters =30;
    pClassifiers(iSubDir) = train_patch_classifier(conf,makeCluster(X,[]),faces.train_faces(~imageData.train.labels),'suffix',...
        subDirs{iSubDir},'override',true,'w1',1,'C',.01);    
    imshow(showHOG(conf,pClassifiers(iSubDir)));    
    pause
end

X_test = imageSetFeatures2(conf,faces.test_faces,true,conf.features.winsize*8);
X_test_2 = imageSetFeatures2(conf,flipAll(faces.test_faces),true,conf.features.winsize*8);

iClassifier = 1;

for iClassifier = 1:3
scores =  max(X_test'*[pClassifiers(iClassifier).w] , X_test_2'*[pClassifiers(iClassifier).w] );
scores = scores+ (imageData.test.faceScores > -.7)';    




[prec,rec,aps] = calc_aps2(scores,imageData.test.labels);
imwrite(showSorted(faces.test_faces,scores,100),fullfile(outDir,[subDirs{iClassifier} '_face_hog.jpg']));
pause;
close all
end


%% make and save enlarged images of the faces...
clear faces_large;
for k = 1:length(imageData.train.imageIDs)    
    k
    I = getImage(conf,imageData.train.imageIDs{k});
    curRect = imageData.train.faceBoxes(k,:);
    curRect = inflatebbox(curRect,1.5);
    faces_large.train_faces{k} = im2uint8(cropper(I,round(curRect)));   
end

for k = 1:length(imageData.test.imageIDs)    
    k
    I = getImage(conf,imageData.test.imageIDs{k});
    curRect = imageData.test.faceBoxes(k,:);
    curRect = inflatebbox(curRect,1.5);
    faces_large.test_faces{k}= cropper(I,round(curRect));    
end

save('/home/amirro/mircs/experiments/common/faces_cropped_new_large.mat','faces_large');

% next: select samples from each of the categories and train a mini-dpm or
% at least a hog classifier; this should be done for slightely extended
% face images to remove border effects.

cur_t = imageData.train.labels;
faceImages = {};
for k = 1:length(cur_t)    
    if (cur_t(k))
        k
        I = getImage(conf,imageData.train.imageIDs{k});
        curRect = imageData.train.faceBoxes(k,:);
%         curRect = makeSquare(curRect);
        curRect = inflatebbox(curRect,1.5);        
        faceImages{end+1} = cropper(I,round(curRect));
    end
end

mImage(faceImages);
multiWrite(faceImages,fullfile(outDir,'faces_1_5'));


% subDirs = {'drink_side','cup_front','cup_side'};
% pClassifiers_sub = makeCluster(zeros(1984,1),[]);
% iSubDir = 1;
% for iSubDir = 1:length(subDirs)
%     curSubDir = fullfile(outDir,'faces',subDirs{iSubDir});    
%     %imgs = dir(fullfile(curSubDir,'*.jpg'));    
%     flippedSubDir = fullfile(curSubDir,'flipped');
%     m = multiRead(conf,flippedSubDir);      
    M1 = multiRead(conf,fullfile(outDir,'faces_1_5','side'));
    rects = selectSamples(conf,M1,fullfile(outDir,'faces_1_5','side','samples'));
    
    rects = cat(1,rects{:});
    rects = makeSquare(imrect2rect(rects));   
    mm = multiCrop(conf,M1,rects,[64 64]);
    toFlip = [1 3 4 6 12 14 17 19 23];
    mm(toFlip) = flipall(mm(toFlip));
    mImage(mm);
    
    conf.features.winsize = [8 8];
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    X = imageSetFeatures2(conf,mm,true,conf.features.winsize*conf.features.vlfeat.cellsize);
    conf.clustering.num_hard_mining_iters =30;
    pClassifiers_sub(iSubDir) = train_patch_classifier(conf,makeCluster(X,[]),faces.train_faces(~imageData.train.labels),'suffix',...
        subDirs{iSubDir},'override',true,'w1',1,'C',.01);    
%     imshow(showHOG(conf,pClassifiers_sub(iSubDir).w))
    pause
% end

conf.get_full_image = false;
% get lip images.
cur_t = imageData.test.labels;
faceImages = {};
for k = 1:length(cur_t)    
    if (cur_t(k))
        k
        I = getImage(conf,imageData.test.imageIDs{k});      
        curRect = imageData.test.faceBoxes(k,:);
        curRect = makeSquare(curRect);
        curRect = inflatebbox(curRect,1.2);        
        faceImages{end+1} = cropper(I,round(curRect));
    end
end
%faceImages2 = cellfun2(@(x) imResample(x(end/3:end,:,:),2*[64*2/3 64],'bilinear'),faceImages);
% faceImages2 = cellfun2(@(x) imResample(x(end/4:end,:,:),2*[48 64],'bilinear'),faceImages);

faceImages2 = cellfun2(@(x) imResample(x,2*[64 64],'bilinear'),faceImages);
conf.detection.params.detect_levels_per_octave = 8;
conf.detection.params.detect_min_scale =1 ;
conf.detection.params.detect_add_flip = 1;
%qq = applyToSet(conf,pClassifiers_sub(1),faceImages2,[],'drink_sub_2','override',true);

qq = applyToSet(conf,pClassifiers_sub(1),faces_large.test_faces,[],'drink_sub_2','override',true);


newScores = scores;

scores2 = -5*ones(size(newScores));
clocs = qq.cluster_locs;
for k = 1:size(clocs,1)
    ind = clocs(k,11);
    scores2(ind) = clocs(k,12);
end

%%
scores =  max(X_test'*[pClassifiers(iClassifier).w] , X_test_2'*[pClassifiers(iClassifier).w] );
scores = scores+ (imageData.test.faceScores > -.8)';    
scores3 = scores+.1*scores2 +...
    ((L1.test_saliency.stds+L1.test_saliency.means_inside-L1.test_saliency.means_outside)>.2)';
scores3 = scores3;
scores3(isnan(scores3)) = -1;
% scores3 = scores;

scores3 =((L1.test_saliency.stds+L1.test_saliency.means_inside+0*L1.test_saliency.means_outside))'+...
    (imageData.test.faceScores > -.6)';
scores3(isnan(scores3)) = -1000;
[prec,rec,aps] = calc_aps2(scores3,imageData.test.labels);
%%
showSorted(faces.test_faces,scores3,100);

% conf.suffix = 'rgb';
% dict = learnBowDictionary(conf,train_faces,true);
% model.numSpatialX = [2];
% model.numSpatialY = [2];
% model.kdtree = vl_kdtreebuild(dict) ;
% model.quantizer = 'kdtree';
% model.vocab = dict;
% model.w = [] ;
% model.b = [] ;
% model.phowOpts = {'Color','RGB'};
names = {'drinking','smoking','blowing_bubbles','brushing'};
classes_of_interest = [conf.class_enum.DRINKING,...
    conf.class_enum.SMOKING,...
    conf.class_enum.BLOWING_BUBBLES,...
    conf.class_enum.BRUSHING_TEETH];
   
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
trainData = extractSamples(conf,imageData.train,faces.train_faces,classes_of_interest,all_train_labels)
testData = extractSamples(conf,imageData.test,faces.test_faces,classes_of_interest,all_test_labels)

conf.features.vlfeat.cellsize = 8;
X_train = imageSetFeatures2(conf,trainData.imgs,true,[]);
X_test = imageSetFeatures2(conf,testData.imgs,true,[]);

close all;
svm_models = {};

for k = 1:4
% k=1
%     k = 2
    cur_y_train = 2*(trainData.labels==k)-1;
    cur_y_test = 2*(testData.labels==k)-1;
    ss2 = ss;
    ss2 = '-t 2 -c .1 w1 1';
    cur_svm_model= svmtrain(cur_y_train, double(X_train'),ss2);
    svm_models{k} = cur_svm_model;
    [predicted_label, ~, cur_model_res] = svmpredict(cur_y_test,double(X_test'),cur_svm_model);
    
%     w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;    
%     cur_model_res = w'*X_test;
    
    %     TT = .1;
    %     cur_model_res = normalise(cur_model_res);
    %     cur_model_res(cur_y_test==1)= cur_model_res(cur_y_test==1).*(rand(1,sum(cur_y_test==1)) >= TT);
    %
    %     plot(cur_svm_model.sv_coef)
    
    [prec,rec,aps] = calc_aps2(cur_model_res,cur_y_test==1);
    figure(1),plot(rec,prec); 
    title([names{k} ' : '  num2str(aps)]);
   % [r,ir] = sort(cur_model_res,'descend');
   showSorted(testData.imgs,cur_model_res,50);
    %figure(2),imshow(multiImage(test_images(ir(1:50)),false));   
    title([names{k} ' : '  num2str(aps)]);
    pause;
end

% psix_train_local = getBOWFeatures(conf,model,trainData.imgs);
end

function d = extractSamples(conf,imageData,faces,classes_of_interest,all_labels)
    imgs = {};
    labels = [];   
    for iClass = 1:length(classes_of_interest)
        f = find(all_labels==classes_of_interest(iClass));
        f_ = cellfun2(@(x) imResample(im2single(x),[64 64],'bilinear'), faces(f));
        
        imgs = [imgs,f_];
        labels = [labels;iClass*ones(length(f),1)];
    end
    d.imgs = imgs;
    d.labels = labels;
end