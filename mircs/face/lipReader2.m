initpath;
config;
load lipData.mat;
close all;

initLandMarkData;

% lipImages_train = lipImages_train2;
% lipImages_test = lipImages_test2;
% 
posLipImages= lipImages_train(t_train);
posInds = find(t_train);
ignoreList = false(size(posLipImages));
% figure,imshow(multiImage(posLipImages));

% drinking from straw
f = find(~ignoreList); 
strawInds = [1 5 6 14 18 19 21 23 27 31 38 42 46 51 54];
ignoreList (strawInds) = true;
strawInds_abs = posInds(strawInds);
posInds(ignoreList) = [];
makeSpecializedFunction('straw');

figure,imshow(multiImage(posLipImages(f(strawInds))))
mImage(lipImages_train(strawInds_abs));

strawSubImages = posLipImages(f(strawInds));

f = find(~ignoreList); 
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 7 8 18 21 27 36 37 43 44 45 47];
cupInds_1 = f(m);
cupInds_1_abs = posInds(m);
ignoreList(cupInds_1) = true;
posInds(m) = [];
makeSpecializedFunction('cup_1'); % bottom cup
figure,imshow(multiImage(posLipImages(cupInds_1)))
cup1SubImages = posLipImages(cupInds_1);
mImage(lipImages_train(cupInds_1_abs));

f = find(~ignoreList);
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 6 11 13 22 25 26];
cupInds_2 = f(m);
cupInds_2_abs = posInds(m);
ignoreList(cupInds_2) = true;
posInds(m) = [];
makeSpecializedFunction('cup_2'); % side cup/ can with hand
figure,imshow(multiImage(posLipImages(cupInds_2)))
mImage(lipImages_train(cupInds_2_abs));
cup2SubImages = posLipImages(cupInds_2);

f = find(~ignoreList); figure,imshow(multiImage(posLipImages(~ignoreList)));
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 5 6 12 13 15 20 24 26 27 28];
cupInds_3 = f(m);
cupInds_3_abs = posInds(m);
ignoreList(cupInds_3) = true;
posInds(m) = [];
makeSpecializedFunction('cup_3'); % side cup/ can with hand
figure,imshow(multiImage(posLipImages(cupInds_3)))
mImage(lipImages_train(cupInds_3_abs));
cup3SubImages = posLipImages(cupInds_3);

f = find(~ignoreList);
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [7 10 17];
bottleInds = f(m);
bottleInds_abs = posInds(m);
ignoreList(bottleInds) = true;
posInds(m) = [];
makeSpecializedFunction('bottle'); % side bottle/ can with hand
figure,imshow(multiImage(posLipImages(bottleInds)))
mImage(lipImages_train(bottleInds_abs));
bottleSubImages = posLipImages(bottleInds);

conf.features.vlfeat.cellsize = 8;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
conf.clustering.num_hard_mining_iters = 12;
conf.detection.params.detect_keep_threshold = -1;

close all;


%% straws.
mImage(lipImages_train(strawInds_abs));

 load ~/storage/train_gpbs.mat
 %%
f = find(t_train);

for k = 1:length(f)
    ii = f(k);
    clf; subplot(1,2,1);
    imshow(train_faces{ii});
    hold on; 
    plotBoxes2(curLipBox([2 1 4 3]),'g');
    subplot(1,2,2);
    edges = train_gpbs(ii).gPb_thin;
    edges = edges(64:end-63,64:end-63);
    curLipBox = lipBoxes_train(ii,:);
    imagesc(edges>0);axis image;
    
    pause
end
 

