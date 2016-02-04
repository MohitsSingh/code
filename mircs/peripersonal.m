
% explore the idea of predicting where to search for action mircs given an
% image.
initpath;
config;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
poseletResultDir = 'poselets_quick';
%%
model = load('/home/amirro/code/3rdparty/poselets_matlab_april2013/data/person/model.mat');
addpath(genpath('/home/amirro/code/3rdparty/sc/'));
conf.poseletResultsDir = poseletResultDir;
conf.poseletModel = model;

% waving hands,
% drinking,
% telescope,
% reading,
% texting message,...
% taking photos,
% using a computer,
% pouring liquid,
% brushing teeth
% washing dishes,
% phoning,
% pushing a cart,
% fixing a bike,
% gardening,
% applauding,
% cutting vegetables,
% blowing bubbles
% fishing,
% looking throught a microscope

% cooking, writing on book, shooting arrow, writing on board, holding
% umbrella

% running

% cooking - in progress


 non_transitive_actions = [conf.class_enum.APPLAUDING;...
     conf.class_enum.CLIMBING;conf.class_enum.JUMPING;...
     conf.class_enum.RUNNING;conf.class_enum.WAVING_HANDS];

%%
C = conf.class_enum;
my_classes = [C.BLOWING_BUBBLES,C.BRUSHING_TEETH,C.DRINKING,C.LOOKING_THROUGH_A_MICROSCOPE,C.LOOKING_THROUGH_A_TELESCOPE,...
    C.PHONING,C.SMOKING,C.TAKING_PHOTOS];

for t = my_classes


    A{t}
    if (ismember(t,non_transitive_actions))
        continue
    end
    t
    conf.class_subset = t;
    [action_rois,true_ids] = markActionROI(conf);
end
%%
% [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');

u = 0;
conf.get_full_image = true
for t = 1:length(true_ids)
        u = u+1;
        I = getImage(conf,true_ids{t});
        clf; imagesc2(I); plotBoxes(action_rois(u,:));
        pause  
end




[poseletPreds,allProbs] = learnBoxPredictions(conf,action_rois);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

%%

suffix = 'straw';
detector = train_patch_classifier(conf,[],[],...
    'suffix',suffix,'toSave',true,'override',false,'C',1);

conf.features.winsize = [7 7];

% run the "bottle in lips" detector in high-probability regions.
q = 0;
conf.get_full_image = true;
conf.detection.params.detect_max_windows_per_exemplar = inf;
conf.detection.params.detect_keep_threshold = -inf;
allClusterLocs = {};
for k = 1:length(test_ids)
    k
    if (~test_labels(k))
        continue;
    end
    q = q+1;
    [ I,xmin,xmax,ymin,ymax ] = getImage(conf,test_ids{k});
    clf; subplot(1,2,1);
    imagesc(I); axis image
    subplot(1,2,2); imagesc(allProbs{q}); axis image    
    curIm = cropper(I,[xmin ymin xmax ymax]);
    curProb = normalise(cropper(allProbs{q},[xmin ymin xmax ymax]));            
    conf.detection.params.max_models_before_block_method = 0;    
    q_current = applyToSet(conf,detector,{curIm},[],...
    [ 'nos_t_face'],'useLocation',{curProb},'disp_model',true,'override',false,'toSave',false);
    allClusterLocs{k} = q_current.cluster_locs;
%     pause;

%     break;
end

for z = 1:length(allClusterLocs)
    if (~isempty(allClusterLocs{z}))
        allClusterLocs{z}(11) = z;
    end
end
aaa = cat(1,allClusterLocs{:});

bbb= visualizeLocs2_new(conf,test_ids,aaa);
mImage(bbb);

