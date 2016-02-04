function [model,hard_negs] = do_training(globalOpts,train_images,model,train_iter)
% train a model for train_iter iterations. Each iteration uses the
% previous' iterations top scoring negatives as hard examples.

if (nargin < 4)
    train_iter = 1;
end


class_subset = globalOpts.class_subset;
nClasses = length(class_subset);
nVocClasses = globalOpts.VOCopts.nclasses;

% featSize = getFeatSize(globalOpts);

% z = hkm(zeros(featSize,1));
% model.w = zeros(length(z),nVocClasses);
% model.b = zeros(1,nVocClasses);
hard_negs = cell(1,nVocClasses);
% start from last iteration and make sure the results exist.
modelPath = getModelPath(globalOpts,train_iter);
if (exist(modelPath,'file'))
    load(modelPath);  % we're done with the model
else
    % start from previous iteration and make sure the results exist.
    % get previous hard negatives
    if (train_iter == 1) %treat first iteration differently...
        prev_hard_negs = cell(1,nVocClasses); % no hard negatives yet for first iteration.
        % append hard negatives from previous iteration to the current one.
    else
        % get hard negatives from previous example.
        [~,prev_hard_negs] = do_training(globalOpts,train_images,...
            model,train_iter-1);
    end
    
    % obtain "regular" training instances
    
    for iClass = 1:nClasses
        cls = class_subset(iClass);
        
        
        %if (~exist(instancesPath,'file'))
        scales = [4];%[2 4 6 8 10];
        T = {};
        for iScale = 1:length(scales)
            instancesPath_aib = fullfile(globalOpts.expPath,...
                ['instances0_aib_' num2str(scales(iScale)) '.mat']);
            if (~exist(instancesPath_aib,'file'))
                instancesPath = fullfile(globalOpts.expPath,...
                    ['instances0_' num2str(scales(iScale)) '.mat']);
                
                aib_path = fullfile(globalOpts.expPath,...
                    ['aibmap_' num2str(scales(iScale)) '.mat']);
                if (~exist(aib_path,'file'))
                    if (~exist(instancesPath,'file'))
                        globalOpts.scale_choice = scales(iScale);
                        if (isfield(globalOpts,'map'))
                            rmfield(globalOpts,'map');
                        end
                        trainingInstances = getTrainingInstances(globalOpts,train_images,model,cls);
                        save(instancesPath,'trainingInstances');
                    else
                        load(instancesPath);
                    end                   
                    [parents] = do_aib(trainingInstances,globalOpts);
                    save(aib_path,'parents');
                else
                    load(aib_path);
                end
                
                [cut,map] = vl_aibcut(parents,globalOpts.aib_cut);
                globalOpts.map = map;
                trainingInstances = getTrainingInstances(globalOpts,train_images,model,cls);
                save(instancesPath_aib,'trainingInstances');
            else
                load(instancesPath_aib);
            end
            
            T{iScale} = trainingInstances;
        end
        
        trainingInstances = struct('classID',{[]},...
            'posFeatureVecs',{[]},'negFeatureVecs',{[]});
        for iScale = 1:length(scales)
            trainingInstances.posFeatureVecs = [trainingInstances.posFeatureVecs;...
                T{iScale}.posFeatureVecs];
            trainingInstances.negFeatureVecs = [trainingInstances.negFeatureVecs;...
                T{iScale}.negFeatureVecs];
        end
        clear T;
                                
        % add hard negatives
        trainingInstances.negFeatureVecs = [trainingInstances.negFeatureVecs,prev_hard_negs{cls}];
        m = trainClassifier(globalOpts,trainingInstances);
        if (iClass == 1)
            model.w = zeros(size(m.w,1),nVocClasses);
            model.b = zeros(1,nVocClasses);
        end
        %         model(cls).model = m;
        model.w(:,cls) = m.w;
        model.b(cls) = m.b;
    end
    
    save(modelPath,'model');
end

% check if hard negs needed
if (nargout == 2)
    
    %     applyModel(globalOpts,model,train_images,train_iter);
    iter_suffix = ['train_' num2str(train_iter)];
    globalOpts.debug = 0;
    globalOpts.removeOverlappingDetections = false;
    
    negsPath = fullfile(globalOpts.expPath,sprintf('negs_%03.0f.mat',train_iter));
    
    if (exist(negsPath,'file'))
        load(negsPath);
    else
        
        % so collect hard_negs from previous as well.
        if (train_iter > 1)
            [~,hard_negs] = do_training(globalOpts,train_images,...
                model,train_iter-1);
        else
            hard_negs = cell(1,nVocClasses);
        end
        
        for iClass = 1:nClasses
            cls = class_subset(iClass);
            classID = globalOpts.VOCopts.classes{cls};
            fPath = sprintf(globalOpts.VOCopts.detrespath,...
                [globalOpts.exp_name '_comp3' iter_suffix],classID);
            [ids,t] = textread(sprintf(globalOpts.VOCopts.imgsetpath,...
                [classID '_' globalOpts.VOCopts.trainset]),'%s %d');
            %         ids = ids(1:globalOpts.selection:end);
            %         t = t(1:globalOpts.selection:end);
            
            
            do_old_mining = 1;
            if (do_old_mining)
                ids_neg = ids(t~=1);
                if (~exist(fPath,'file'))
                    applyModel(globalOpts,model,ids_neg,train_iter); %TODO - return
                    % this line to the original location, it being there is a bug
                    
                    collectResults(globalOpts,ids_neg,cls,1,iter_suffix,train_iter);
                    
                end
                fid = fopen(fPath);
                A = textscan(fid,'%s %f %f %f %f %f %f %f');
                fclose(fid);
                ids = A{1};
                boxes =[A{3:end}];
                scores = A{2};
                nSamples = length(ids);
                hardNegs = 1:nSamples;
                ids = ids(hardNegs);
                boxes = boxes(hardNegs,[2 1 4 3]);
                % append previous hard negatives to current hard negatives
                scales = 4;
                iScale = 1;
                aib_path = fullfile(globalOpts.expPath,...                
                    ['aibmap_' num2str(scales(iScale)) '.mat']);
                load(aib_path);
                [cut,map] = vl_aibcut(parents,globalOpts.aib_cut);
                globalOpts.map = map;
                scale = 1;
                hard_negs{cls} = [hard_negs{cls},get_box_features2(globalOpts,boxes,ids,model,scale)];
            else % collect a high-scoring false-positive from some positive images!
                ids_neg = ids(t==1);
                if (~exist(fPath,'file'))
                    applyModel(globalOpts,model,ids_neg,train_iter); %TODO - return
                    % this line to the original location, it being there is a bug?
                    collectResults(globalOpts,ids_neg,cls,0,iter_suffix,train_iter);
                end
                fid = fopen(fPath);
                A = textscan(fid,'%s %f %f %f %f %f');
                fclose(fid);
                ids = A{1};
                boxes =[A{3:end}];
                scores = A{2};
                
                hard_neg_boxes = [];
                hard_neg_ids = {};
                
                for iID = 1:length(ids_neg)
                    currentID = ids_neg{iID};
                    s = find(strncmp(currentID,ids,length(currentID)));
                    scores_ = scores(s);
                    [~,iss] = sort(scores_,'descend');
                    boxes_ = boxes(s(iss,:),:);
                    rec = PASreadrecord(getRecFile(globalOpts,currentID));
                    clsinds=strmatch(classID,{rec.objects(:).class},'exact');
                    isDifficult = [rec.objects(clsinds).difficult];
                    isTruncated = false(size([rec.objects(clsinds).truncated]));%this effectively
                    % toggles usage of truncated examples. 'false' means they are used.
                    
                    % get bounding boxes
                    gtBoxes = cat(1,rec.objects(clsinds(~isDifficult & ~isTruncated)).bbox);
                    %difficultBoxes = cat(1,rec.objects(clsinds(isDifficult)).bbox);
                    
                    overlaps = boxesOverlap(boxes_,gtBoxes);
                    
                    isFalse = find(sum(overlaps > .3,2)==0,1,'first');
                    
                    if (~isempty(isFalse))
                        % show the false positives
                        % visualize...
                        %                         clf;
                        %                         imshow(getImageFile(globalOpts,currentID));
                        %                         hold on;
                        %                         plotBoxes2(boxes_(isFalse,[2 1 4 3]),'Color','g');
                        %                         pause;
                        hard_neg_boxes = [hard_neg_boxes;boxes_(isFalse,[2 1 4 3])];
                        hard_neg_ids = [hard_neg_ids;repmat({currentID},length(isFalse),1)];
                    end%
                end
                
                
                
                %                 nSamples = length(ids);
                %                 hardNegs = 1:nSamples;
                %                 ids = ids(hardNegs);
                %                 boxes = boxes(hardNegs,[2 1 4 3]);
                % append previous hard negatives to current hard negatives
                scale = 1;
                hard_negs{cls} = [hard_negs{cls},get_box_features2(globalOpts,hard_neg_boxes,hard_neg_ids,model,scale)];
                
            end
            
        end
        save(negsPath,'hard_negs');
    end
end

