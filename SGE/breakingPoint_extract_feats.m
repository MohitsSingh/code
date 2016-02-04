function res = breakingPoint_extract_feats(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd /home/amirro/code/breakingPoint;
    init;
    load gt_data;
    res.conf = conf;
    res.train_gt = train_gt;
    res.val_gt = val_gt;
    res.VOCopts = VOCopts;
    return;
end


cls = params.class;
conf = initData.conf;
VOCopts = initData.VOCopts;
train_gt = initData.train_gt;
val_gt = initData.val_gt;
curScale = params.scale;
% get all bounding boxes for a specific class, e.g, horse.

[train_box_data] = getClassBoundingBoxes(train_gt,cls);
pos_train_img_inds = [train_box_data.image_ind];
neg_train_img_inds = setdiff(1:length(train_gt),pos_train_img_inds);
pos_train_imgs = getPosSubImgs(VOCopts, train_gt,train_box_data);
[val_box_data] = getClassBoundingBoxes(val_gt,cls);
pos_val_img_inds = [val_box_data.image_ind];
neg_val_img_inds = setdiff(1:length(val_gt),pos_val_img_inds);
pos_val_imgs = getPosSubImgs(VOCopts, val_gt,val_box_data);

% pos_imgs_orig = pos_imgs;
myNormalizeFun = @(x) normalize_vec(x);
%experiment_results = struct('maxScale',{},'performance',{});
res = struct('maxScale',{},'pos_train',{},'neg_train',{},'pos_val',{},'neg_val',{});
batchSize = 128;
resizeImgs = @(u) cellfun2(@(x) imResample(x, min(1,curScale/size(x,1)),'bilinear'),u);
pos_train_imgs_scaled = resizeImgs(pos_train_imgs);
feats_train_pos = extractDNNFeats(pos_train_imgs_scaled,conf.net,batchSize);
feats_train_neg = extractDNNFeatsForSet(VOCopts,train_gt(neg_train_img_inds),conf.net,curScale);
pos_val_imgs_scaled = resizeImgs(pos_val_imgs);
feats_val_pos = extractDNNFeats(pos_val_imgs_scaled,conf.net,batchSize);
feats_val_neg = extractDNNFeatsForSet(VOCopts,val_gt(neg_val_img_inds),conf.net,curScale);
res(1).scale = curScale;
res.pos_train = feats_train_pos;
res.neg_train = feats_train_neg;
res.pos_val = feats_val_pos;
res.neg_val = feats_val_neg;


clear classification_results;
val_feats = [res.pos_val res.neg_val];
for itrial = 1:params.nTrials
    classifier = train_classifier_pegasos(res.pos_train,res.neg_train,0,false);
    scores  = classifier.w(1:end-1)'*val_feats;
    clear r;
    [r.recall, r.precision, r.info] = vl_pr([ones(size(pos_val_imgs)),-ones(size(neg_val_img_inds))],scores);
    classification_results(itrial) = r;
end

res.classification_results = classification_results;