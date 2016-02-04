function x = extractDNNFeatsHelper(conf,imgs,net)
x = {};
batchSize = 100 ;
for t=1:batchSize:length(imgs)
    t/length(imgs)
    batch = imgs(t:min(t+batchSize-1, length(imgs)));
    for tt = 1:length(batch)
        batch{tt} = im2uint8(getImage(conf,batch{tt}));
    end
    x = extractDNNFeats(batch,net);
end
x = cat(2,x{:});
end