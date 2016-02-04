
function feats = extractDNNFeatsForSet(VOCopts,train_gt,net,curScale)
batchSize = 128;
feats = {};
for b = 1:batchSize:length(train_gt)
    batchImgs = {};
    for bb = b:min(b+batchSize-1,length(train_gt))
        batchImgs{end+1} = imread(sprintf(VOCopts.imgpath,train_gt(bb).filename(1:end-4)));
    end
    batchImgs = cellfun2(@(x) imResample(x, min(1,curScale/size(x,1)),'bilinear'),batchImgs);
    feats{end+1} = extractDNNFeats(batchImgs,net,batchSize);
end

feats = cat(2,feats{:});