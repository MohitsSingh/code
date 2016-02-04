% detect_interaction_seg

if (~exist('toStart','var'))
    initpath;
    config;
    load lipData.mat;    
    initLandMarkData;
    % %     close all;
    clf;    
    conf.class_subset = conf.class_enum.DRINKING;
    soStart = 1;
end

%f = find(t_train)+100;
f = randperm(length(t_train));
conf.get_full_image = true;
%%
close all;



initpath; 
config;
conf.class_subset = conf.class_enum.DRINKING;
roiPath = '~/storage/action_rois';
[action_rois,true_ids,all_ids,labels] = markActionROI(conf,roiPath);
action_rois_poly = actionRoisToTrainData(action_rois,true_ids);
conf.featConf = init_features(conf);
conf.get_full_image = true;


drink_bogus = learnModels(conf,train_ids,train_labels,action_rois_poly,{'all'},'drink_bogus');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
drinking_test = test_ids(test_labels);

for k = 19:100
    currentID = drinking_test{k};
    [regions,regionConfs] = applyModel(conf,currentID,drink_bogus.models(1));
    pause;
%     displayRegions(getImage(conf,currentID),regions);
  
end
% profile viewer;


for k = 1:length(f)
    clf;
    currentID = train_ids_r{f(k)};
    gpbFile = fullfile(conf.gpbDir,strrep(currentID,'.jpg','.mat'));
    regionsFile = strrep(gpbFile,'.mat','_regions.mat');

    load(regionsFile);
    I = getImage(conf,currentID);
    regions = fillRegionGaps(regions);
    T_ovp = .8;
    regionSel = suppresRegions(regionOvp, T_ovp);
    regions = regions(regionSel);
    regionOvp = regionOvp(regionSel,regionSel);
    
    [~,rr] = imcrop(I);
    rr = imrect2rect(rr);
    ovp = boxRegionOverlap(rr,regions,dsize(I,1:2));
    
    [o,io] = sort(ovp,'descend');
    displayRegions(im2double(I),regions,io(1:5));
    
end