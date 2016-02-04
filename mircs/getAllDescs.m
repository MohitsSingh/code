function all_descs = getAllDescs(conf,model,imgs,sz,cacheFileName)
if (cacheFileName(1) == '-')
    cacheFileName = cacheFileName(2:end);
    delete(cacheFileName,'file');
end

if (exist(cacheFileName,'file'))
    load(cacheFileName);
    return;
end

all_descs = struct('frames',{},'descrs',{},'HOG',{});

for k = 1:length(imgs)
    k
    im = getImage(conf,imgs{k});
    if (~isempty(sz))
        im = imresize(im,sz,'bilinear');
    end
    [frames, descrs] = vl_phow(im, model.phowOpts{:});
    %TODO - this assumes a max image size of 2^16 (which is very
    %reasonable)
    all_descs(k).frames = uint16(frames(1:2,:));
    
    %TODO - this assumes a max dictionary of 2^16
    binsa = uint16(vl_kdtreequery(model.kdtree, model.vocab, ...
        single(descrs), ...
        'MaxComparisons', 15)) ;
    all_descs(k).binsa = binsa;
    all_descs(k).descrs = [];
    %     im = im2single(im);
    %     all_descs(k).HOG = vl_hog(im,conf.features.vlfeat.cellsize,'NumOrientations',9);
end

save(cacheFileName,'all_descs','-v7.3');
end