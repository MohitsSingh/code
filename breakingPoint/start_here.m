if (~exist('initialized','var') || ~initialized)
    init;
    if (exist('gt_data.mat','file'))
        load('gt_data');
    else
        train_gt = loadGT(VOCopts,'train');
        val_gt = loadGT(VOCopts,'val');
        save('gt_data.mat','train_gt','val_gt');
    end
    initialized = true;
end

debug_ = false;

% get all bounding boxes for a specific class, e.g, horse.
%cls = 'horse';
for iClass = 1:length(VOCopts.classes)
    cls = VOCopts.classes{iClass};
    % cls = 'bottle';
    
    train_gt = train_gt(1:1:end);
    val_gt = val_gt(1:1:end);
    
    [train_box_data] = getClassBoundingBoxes(train_gt,cls);
    pos_train_img_inds = [train_box_data.image_ind];
    neg_train_img_inds = setdiff(1:length(train_gt),pos_train_img_inds);
    pos_train_imgs = getPosSubImgs(VOCopts, train_gt,train_box_data);
    
    [val_box_data] = getClassBoundingBoxes(val_gt,cls);
    pos_val_img_inds = [val_box_data.image_ind];
    neg_val_img_inds = setdiff(1:length(val_gt),pos_val_img_inds);
    pos_val_imgs = getPosSubImgs(VOCopts, val_gt,val_box_data);
    
    maxScale = [3:25 30 35 40 50 60 80 100 150 200];
    
    % pos_imgs_orig = pos_imgs;
    myNormalizeFun = @(x) normalize_vec(x);
    %experiment_results = struct('maxScale',{},'performance',{});
    
    experiment_feats = struct('maxScale',{},'pos_train',{},'neg_train',{},'pos_val',{},'neg_val',{});
    batchSize = 128;
    for iScale = 1:length(maxScale)
        iScale
        curScale = maxScale(iScale);
        
        curPath = sprintf('~/storage/misc/data/scale_%s_%03.0f.mat',cls,curScale);
        if (exist(curPath,'file'))
            %         load(curPath);
            %         experiment_feats(iScale) = curScaleStuff;
            continue;
        end
        
        resizeImgs = @(u) cellfun2(@(x) imResample(x, min(1,curScale/size(x,1)),'bilinear'),u);
        pos_train_imgs_scaled = resizeImgs(pos_train_imgs);
        feats_train_pos = extractDNNFeats(pos_train_imgs_scaled,conf.net,batchSize);
        feats_train_neg = extractDNNFeatsForSet(VOCopts,train_gt(neg_train_img_inds),conf.net,curScale);
        
        pos_val_imgs_scaled = resizeImgs(pos_val_imgs);
        feats_val_pos = extractDNNFeats(pos_val_imgs_scaled,conf.net,batchSize);
        feats_val_neg = extractDNNFeatsForSet(VOCopts,val_gt(neg_val_img_inds),conf.net,curScale);
        
        experiment_feats(iScale).maxScale = curScale;
        experiment_feats(iScale).pos_train = feats_train_pos;
        experiment_feats(iScale).neg_train = feats_train_neg;
        experiment_feats(iScale).pos_val = feats_val_pos;
        experiment_feats(iScale).neg_val = feats_val_neg;
        
        curScaleStuff = experiment_feats(iScale);
        save(curPath,'curScaleStuff');                        
    end
    
    % save experiment_feats experiment_feats
    
    % train/test classifiers...
    %%
    
    classification_results_path = sprintf('%s_classification_results.mat',cls);
    if (exist(classification_results_path,'file'))
        continue;
    else
        clear classification_results;
        
        for scale_ind = 1:length(maxScale)
            curScale = maxScale(scale_ind)
            curPath = sprintf('~/storage/misc/data/scale_%s_%03.0f.mat',cls,curScale);
            load(curPath);
            %     curScaleStuff = experiment_feats(scale_ind);
            val_feats = [curScaleStuff.pos_val curScaleStuff.neg_val];
            for itrial = 1:10
                classifier = train_classifier_pegasos(curScaleStuff.pos_train,curScaleStuff.neg_train,0,false);                            
                scores  = classifier.w(1:end-1)'*val_feats;
                clear r;
                [r.recall, r.precision, r.info] = vl_pr([ones(size(pos_val_imgs)),-ones(size(neg_val_img_inds))],scores)
                r.scale = curScale;
                classification_results(itrial,scale_ind) = r;
            end
        end
        
        save(classification_results_path,'classification_results');
    end
end
%%


A = {};b = false;
for iScale = 1:length(maxScale)
    iScale
    for iClass = 1:length(VOCopts.classes)
        curName = sprintf('cls_%s_scale_%03.0f',VOCopts.classes{iClass},maxScale(iScale));
        p= j2m('~/storage/breaking_point_res',curName);
        if (exist(p,'file'))
            load(p,'classification_results');
            A{iScale,iClass} = classification_results;
            %                break
        end
        %            if (b) break ,end
    end
    %     if (b) break ,end
end


% choose a class...
%%
cls = VOCopts.classes{5}
%
[train_box_data] = getClassBoundingBoxes(train_gt,cls);
pos_train_img_inds = [train_box_data.image_ind];
pos_train_imgs = getPosSubImgs(VOCopts, train_gt,train_box_data);
%%
curScale = maxScale(20)
resizeImgs = @(u) cellfun2(@(x) imResample(x, min(1,curScale/size(x,1)),'bilinear'),u);
pos_train_imgs_scaled = resizeImgs(pos_train_imgs);
mImage(pos_train_imgs_scaled);

% plot(A.classification_results(15).recall,A.classification_results(15).precision)
%%
aps = zeros(size(A));
for ii = 1:size(aps,1)
    for jj = 1:size(aps,2)
        jj
        curA = A{ii,jj};
        if (~isempty(curA))
            [ii jj]
            infos = [curA.info];
            aps(ii,jj) = mean([infos.ap]);
        end
    end
end
imagesc(aps)

%%
I = pos_val_imgs{1};

size(aps)
plot(median(aps,2))
figure,imagesc(aps)
