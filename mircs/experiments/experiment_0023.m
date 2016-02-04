%%%% experiment 23 %%%
%% Feb 2, 2014

%% Run the upper body detector.

if (0)            
    initpath;
    config;
    
    dpmPath = '~/code/3rdparty/voc-release5/';
    addpath(genpath(dpmPath));
    startup;
    load /home/amirro/code/3rdparty/disc_subcat/ubDetModel.mat
    visualizemodel(model)
end

conf.get_full_image = true;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
ids_ = test_ids;
labels_ = test_labels;imageSet = imageData.test;
conf.get_full_image = false;
ids_ = test_ids(test_labels);
is = 1:length(ids_);
% test_scores = -inf*ones(size(ids_));
%%
for ik = 1:length(ids_)
    k
    k = is(ik);
    currentID = ids_{k};
    I = getImage(conf,currentID);
    
    ds = imgdetect(I,model,-1.5);
    
    ds = ds(nms(ds,.5),:);
    if (~isempty(ds))
        clf;showboxes(I,ds(1,:));
    end
    
    pause;
    
    
end