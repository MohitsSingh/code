%% experiment 0045
% 26/6/2014
% train dpm for the fra_db hands, head, objects

if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    classNames = unique(all_class_names);
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    isTrain = [fra_db.isTrain];
    initialized = true;
end

nonPersonIDS = getNonPersonIds(VOCopts);
false_ids = vl_colsubset(nonPersonIDS',100,'uniform');

% only the object models using the major axis for rotation.
models_dir = 'dpm_models_obj';ensuredir(models_dir);
all_models =[];
for iClass = 1:length(classes)
    curClass = classes(iClass);
    isClass = find([fra_db.classID]==curClass);
       
    % objects
    pos_ids = {};
    pos_boxes = {};
    cls = [fra_db(isClass(1)).class '_obj'];
    for t = 1:length(fra_db)
        curImgData = fra_db(t);
        if (~curImgData.isTrain || curImgData.classID~=curClass) , continue,end;
        t
        objects = curImgData.objects;
        for iObject = 1:length(objects)
            
            curPoly = objects(iObject).poly;
            b = poly2mask2(curPoly,curImgData.size);
            b = bwmorph(b,'clean');
            rprops = regionprops(b,'Orientation');            
            pos_ids{end+1} = imrotate(getImage(conf,curImgData.imageID),-rprops.Orientation,'bilinear');
            b = (imrotate(b,-rprops.Orientation));
%             clf;displayRegions(pos_ids{end},b);pause
            curBox = region2Box(b);            
            pos_boxes{end+1} = curBox;
        end
    end
    
    pos_boxes = cat(1,pos_boxes{:});
    trainSet = prepareForDPM(conf,pos_ids,false_ids,pos_boxes);
    n = 1; % number of subclasses
    valSet = [];
    cls = [cls '_rot'];
    [model,locked] = runDPMLearning(cls, n, trainSet, valSet);
    all_models = [all_models,model];
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
   
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
end



models_dir = 'dpm_models';
ensuredir(models_dir);
all_models =[];
for iClass = 1:length(classes)
    curClass = classes(iClass);
    isClass = find([fra_db.classID]==curClass);
    % hands
    pos_ids = {};
    pos_boxes = {};
    cls = [fra_db(isClass(1)).class '_hand'];
    for t = 1:length(fra_db)
        curImgData = fra_db(t);
        if (~curImgData.isTrain || curImgData.classID~=curClass) , continue,end;
        hands = curImgData.hands;
        for iHand = 1:size(hands,1)
            pos_ids{end+1} = curImgData.imageID;
            pos_boxes{end+1} = hands(iHand,:);
        end
    end
    pos_boxes = cat(1,pos_boxes{:});
    trainSet = prepareForDPM(conf,pos_ids,false_ids,pos_boxes);
    n = 2; % number of subclasses
    valSet = [];
    [model,locked] = runDPMLearning(cls, n, trainSet, valSet);
    all_models = [all_models,model];
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
    
    % objects
    pos_ids = {};
    pos_boxes = {};
    cls = [fra_db(isClass(1)).class '_obj'];
    for t = 1:length(fra_db)
        curImgData = fra_db(t);
        if (~curImgData.isTrain || curImgData.classID~=curClass) , continue,end;
        objects = curImgData.objects;
        for iObject = 1:length(objects)
            pos_ids{end+1} = curImgData.imageID;
            pos_boxes{end+1} = pts2Box(objects(iObject).poly);
        end
    end
    
    pos_boxes = cat(1,pos_boxes{:});
    trainSet = prepareForDPM(conf,pos_ids,false_ids,pos_boxes);
    n = 2; % number of subclasses
    valSet = [];
    [model,locked] = runDPMLearning(cls, n, trainSet, valSet);
    all_models = [all_models,model];
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
    % heads
    pos_ids = {};
    pos_boxes = {};
    cls = [fra_db(isClass(1)).class '_head'];
    for t = 1:length(fra_db)
        curImgData = fra_db(t);
        if (~curImgData.isTrain ||curImgData.classID~=curClass) , continue,end;
        faceBox = curImgData.faceBox;
        pos_ids{end+1} = curImgData.imageID;
        pos_boxes{end+1} = faceBox;
        
    end
    pos_boxes = cat(1,pos_boxes{:});
    trainSet = prepareForDPM(conf,pos_ids,false_ids,pos_boxes);
    n = 2; % number of subclasses
    valSet = [];
    [model,locked] = runDPMLearning(cls, n, trainSet, valSet);
    all_models = [all_models,model];
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
    
    % phraselet (the entire head area + vicinity)
    pos_ids = {};
    pos_boxes = {};
    cls = [fra_db(isClass(1)).class '_phrase'];
    for t = 1:length(fra_db)
        curImgData = fra_db(t);
        if (~curImgData.isTrain ||curImgData.classID~=curClass) , continue,end;
        faceBox = curImgData.faceBox;
        faceBox = round(inflatebbox(faceBox,2.5,'both',false));
        pos_ids{end+1} = curImgData.imageID;
        pos_boxes{end+1} = faceBox;
        
    end
    pos_boxes = cat(1,pos_boxes{:});
    trainSet = prepareForDPM(conf,pos_ids,false_ids,pos_boxes);
    n = 2; % number of subclasses
    valSet = [];
    [model,locked] = runDPMLearning(cls, n, trainSet, valSet);
    all_models = [all_models,model];
    if (~locked)
        save(fullfile(models_dir,cls),'model');
    end
end
save(fullfile(models_dir,'fra_models.mat'), 'all_models');







