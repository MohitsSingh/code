% 25/7/2013
% purpose of this script is to run the hand detection code on the entire
% standford-40 images dataset, for high-recall. 

initpath;
config;
%%
% for q = 1:40
conf.class_subset = q;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');


load('~/storage/hands_s40/top_bbs_train.mat');
inds = {};
for k = 1:length(top_bbs_train)
    inds{k} = k*ones(size(top_bbs_train{k},1),1);
end
conf.get_full_image = 1;
bbs = zeros(length(top_bbs_train),5);
for k = 1:length(top_bbs_train)    
   bbs(k,:) = top_bbs_train{k}(1,[1:4 6]);
end

subsel = 1:4000;
bbs_sub = bbs(subsel,:);
ids_sub = train_ids(subsel);
[s,is] = sort(bbs_sub(:,end),'descend');
sel_ = is(1:10);
conf.not_crop = true;

M = multiCrop(conf,ids_sub(sel_),bbs_sub(sel_,:),[]);
close all; mImage(M);
% title(strrep(conf.classes{q},'_',' '));
% pause;






    

% % get all image paths...
% imPath = getImagePath(conf,train_ids{1});
% 
% 
% 
% % code for parallel running....
% 
% handDetectionBaseDir = '/home/amirro/code/3rdparty/hand_detector_code';


