% cup - detector... shot N!
% mkdir dpm_models
if (~exist('dpm_models/model_drinking_cups.mat','file'))
    initpath;
    config;
    roiPath = '~/storage/cup_rois';
    conf.get_full_image = true;
    % prepare data for dpm...
    conf.class_subset = conf.class_enum.DRINKING;
    [action_rois,true_ids] = markActionROI(conf,roiPath);
    action_rois_s = makeSquare(action_rois);
    % action_rois_s = inflatebbox(action_rois_s,1.1,'both',false);
    
    % driking
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    cup_sel = [2 5 9 14 20 21 23 28 29 41 44 45];
    action_rois_s = action_rois_s(cup_sel,:);
    true_ids  = train_ids(train_labels);
    true_ids = true_ids(cup_sel);
    
    false_ids = train_ids(~train_labels);
    trainSet = prepareForDPM(conf,true_ids,false_ids,action_rois_s);
    cls = conf.classes{conf.class_subset};
    n = 2; % number of subclasses
    valSet = [];
    model_drinking_cups = runDPMLearning(cls, n, trainSet, valSet);
    save dpm_models/model_drinking_cups model_drinking_cups;
else
    load dpm_models/model_drinking_cups;
end

%% now train for the entire drinking dataset, see what you come up with.
if (~exist('dpm_models/model_drinking_all.mat','file'))
    initpath;
    config;
    roiPath = '~/storage/cup_rois';
    conf.get_full_image = true;
    conf.class_subset = conf.class_enum.DRINKING;
    % prepare data for dpm..., dont square them out.
    [action_rois,true_ids] = markActionROI(conf,roiPath);
   % driking
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    true_ids  = train_ids(train_labels);

    
    false_ids = train_ids(~train_labels);
    trainSet = prepareForDPM(conf,true_ids,false_ids,action_rois);
    cls = conf.classes{conf.class_subset};
    n = 1; % number of subclasses
    valSet = [];
    % sort of random
    
    model_drinking_all = runDPMLearning('drinking_all', n, trainSet, valSet);
    
    % TODO: ok, this is terrible, we need to split according to appearance and
    % train a single model!! per appearance class.
    
    save dpm_models/model_drinking_all model_drinking_all;
else
    load dpm_models/model_drinking_all;
end

%%
if (~exist('dpm_models/model_drinking_all_2.mat','file'))
    initpath;
    config;
    roiPath = '~/storage/cup_rois';
    conf.get_full_image = true;
    conf.class_subset = conf.class_enum.DRINKING;
    % prepare data for dpm..., dont square them out.
    [action_rois,true_ids] = markActionROI(conf,roiPath);
    action_rois_s = makeSquare(action_rois);
    action_images = multiCrop(conf,true_ids,round(action_rois_s),[64 64]);       
    conf.features.vlfeat.cellsize = 4;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.clustering.num_hard_mining_iters = 12;
    conf.features.winsize = [10 10 31];
    conf.detection.params.detect_keep_threshold = -1;    
    x = imageSetFeatures2(conf,action_images,true,[40 40]); 
    
    
    if (~exist('cic.mat','file'))
        [C,IC] = vl_kmeans(x,5,'NumRepetitions',100);
        [cup_clusters,ims]= makeClusterImages(action_images,C,IC,x,'drinking_action_images');
         save cic.mat C IC true_ids  false_ids action_rois;
    else
        model_drinking_all_2 = {};
        for k = 1:5        
            load cic.mat;
            n = 1;
            trainSet = prepareForDPM(conf,true_ids(IC==k),false_ids,action_rois(IC==k,:));
            model = runDPMLearning(['drinking_all_' num2str(k)] , n, trainSet, []);
            eval(sprintf('model_drinking_all_%1d=model',k));
            eval(sprintf('save dpm_models/model_drinking_all_%1d model_drinking_all_%1d;',k,k));                        
        end
    end
            
%    % driking
%     [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
%     true_ids  = train_ids(train_labels);
% 
%     
%     false_ids = train_ids(~train_labels);
%     
%     
%     
%     trainSet = prepareForDPM(conf,true_ids,false_ids,action_rois);
%     cls = conf.classes{conf.class_subset};
%     n = 1; % number of subclasses
%     valSet = [];
%     % sort of random
%     
%     model_drinking_all = runDPMLearning('drinking_all', n, trainSet, valSet);
%     
%     % TODO: ok, this is terrible, we need to split according to appearance and
%     % train a single model!! per appearance class.
%     s
%     save dpm_models/model_drinking_all model_drinking_all;
else
%     load dpm_models/model_drinking_all;
end

