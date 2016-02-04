% detect full landmarks with ramanan.

initpath;
config;
conf.max_image_size = inf;
% whoops - this isn't really full, it uses the p146 model.
train_landmarks_full = detect_landmarks(conf,train_ids,1);
save train_landmarks_full.mat train_landmarks_full;
test_landmarks_full = detect_landmarks(conf,test_ids,1);
save test_landmarks_full.mat test_landmarks_full;


conf.max_image_size = inf;
% whoops - this isn't really full, it uses the p146 model.
train_landmarks_full = detect_landmarks(conf,train_ids,1);
save train_landmarks_full.mat train_landmarks_full;
test_landmarks_full = detect_landmarks(conf,test_ids,1);
save test_landmarks_full.mat test_landmarks_full;

% inflate by 2 just to make sure...
train_landmarks_full2 = detect_landmarks(conf,train_ids,2);
save train_landmarks_full2.mat train_landmarks_full2;
test_landmarks_full2 = detect_landmarks(conf,test_ids,2);
save test_landmarks_full2.mat test_landmarks_full2;


% inflate by 2 , run full model (multipie independent);
train_landmarks_full_p = detect_landmarks_p(conf,train_ids,2);
save train_landmarks_full_p.mat train_landmarks_full_p;
test_landmarks_full_p = detect_landmarks_p(conf,test_ids,2);
save test_landmarks_full_p.mat test_landmarks_full_p;


detect_landmarks(conf,train_ids(881),1);

%%
% load train_landmarks_full

load test_landmarks_full


[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');

% show the landmarks for the training images...
[faceLandmarks_train,allBoxes_train] = landmarks2struct(train_landmarks_full2,train_ids,conf);
[faceLandmarks_test,allBoxes_test] = landmarks2struct(test_landmarks_full2,test_ids,conf);


% % 
% % r = find(train_labels);
% % c = length(r);
% % for q = 1:length(r)
% %     c = c-isempty(faceLandmarks_train(r(q)).xy);
% % end
% % % 53/100 faces discovered
% % r = find(test_labels);
% % c = length(r);
% % for q = 1:length(r)
% %     c = c-isempty(faceLandmarks_test(r(q)).xy);
% % end
% % % also, 70/156 faces discovered
% % 
% % r = find(train_labels);
% % c = length(r);
% % for q = 1:length(r)
% %     c = c-isempty(faceLandmarks_train(r(q)).xy);
% % end
% % 
% % % 50% of faces missed



%%

conf.max_image_size = inf;

faceLandmarks = faceLandmarks_train;%(~t_dontcare);
imgSet = train_ids;%(~t_dontcare);
labelSet = train_labels;

% faceLandmarks = faceLandmarks_test;%(~test_dontcare);
% imgSet = test_ids;%(~test_dontcare);
% labelSet = test_labels;

for k = 1:length(imgSet)

    if (~labelSet(k))
        continue;
    end
       
    curImage = getImage(conf,imgSet{k});
    clf;imshow(curImage);
    hold on;
    if (isempty(faceLandmarks(k).xy))
        disp(['no boxes for ' num2str(k)]);
    else
        
        boxesToPlot = cat(1,landmarks_train{k}.xy);
        plotBoxes2(faceLandmarks(k).xy(:,[2 1 4 3])/2);
%         plotBoxes2(boxesToPlot(:,[2 1 4 3])/2);
        pause
    end    
end
   
    
%% refine the boxes
[refined,faceImages] = refineLandmarks(conf,train_ids(train_labels),landmarks_train(train_labels));
%    [landmarks_,lipBoxes,faceBoxes] = landmarks2struct(landmarks_train(train_labels),train_ids(train_labels),conf);
%     faceImages = multiCrop(conf,train_ids(train_labels),round(faceBoxes/2));      

imshow(multiImage(faceImages,false))    
figure,imshow(multiImage(lipImages_train,false))
[lipImages_train_refined,faceScores_train_refined] = getLipImages(conf,faceImages,refined,sz,inflateFactor,2);

figure,imshow(multiImage(lipImages_train,false));
figure,imshow(multiImage(lipImages_train_refined,false)); title('refined');


%%

initpath;
config;
conf.max_image_size = inf;
%use independent model
train_landmarks_full_ind = detect_landmarks(conf,train_ids,1,false);
save train_landmarks_full_ind.mat train_landmarks_full_ind;
test_landmarks_full_ind = detect_landmarks(conf,test_ids,1,false);
save test_landmarks_full_ind.mat test_landmarks_full_ind;


%%
load train_landmarks_full_ind

% train a multiview face detector...
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

conf.max_image_size = inf;

[faceLandmarks,allBoxes_complete,faceBoxes_complete] = landmarks2struct(train_landmarks_full_ind,train_ids,conf);

s = [faceLandmarks.s];
c = [faceLandmarks.c];
[s,is] = sort(s,'descend');
c = c(is);

 %posemap = 90:-15:-90;
posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];

for k = 1:length(s)
    k
    if (~train_labels(is(k)))
        continue;
    end
    im = getImage(conf,train_ids{is(k)});
    clf;imshow(im);
    hold on;
    bs = train_landmarks_full_ind{is(k)};
    showboxes(im,bs(1),posemap);
    pause;
end

% is = 1:10;
%(bbox,inflation,direction,absFlag)
faceBoxes_complete_1 = inflatebbox(faceBoxes_complete,2,'both',false);

faceImages = multiCrop(conf,train_ids(is(1:50)),...
    round(faceBoxes_complete_1(is(1:50),:)));

mImage(faceImages);
faceLandmarks = faceLandmarks(is);
train_ids_s = train_ids(is);
%%


subImages ={};

for iPose =1:length(posemap)
    iPose
    curC = find(c==iPose);
    goodFaces = [faceLandmarks(curC).s] >= -.6;
    if (sum(goodFaces)==0)
        disp(['no good faces found for pose ' num2str(posemap(iPose))]);
        continue;
    end
    curFaceImages = multiCrop(conf,train_ids_s(curC(goodFaces)),...
    round(faceBoxes_complete(is(curC(goodFaces)),:)),[64 64]);
    mImage(curFaceImages);
    subImages{iPose} = curFaceImages;
    pause;
    close all;
end

clusters = makeCluster(0,[]);
conf.max_image_size = inf;
conf.features.vlfeat.cellsize = 8;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;

for k = 1:length(subImages)
    clusters(k) = makeCluster(0,[]);
    clusters(k).isvalid = 0;
    if (~isempty(subImages{k}))        
        x = imageSetFeatures2(conf,subImages{k},true,[]);
        clusters(k) = makeCluster(x,[]);
    end
end

conf.detection.params.max_models_before_block_method = 10;
conf.max_image_size = 256;
conf.clustering.num_hard_mining_iters = 12;
newFaceDetectors = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),...,
    'suffix','newFaceDetectors','C',1,'override',true);

for k = 1:length(newFaceDetectors)
    if (newFaceDetectors(k).isvalid)
        imshow(showHOG(conf,newFaceDetectors(k)));
        pause;
    end
end
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
conf.get_full_image = 0 ;
conf.max_image_size = inf;
faces_train_try = applyToSet(conf,newFaceDetectors,train_ids(train_labels),[],'faces_train_try1','override',true);

faces_train_try(1).cluster_locs(:,12)


