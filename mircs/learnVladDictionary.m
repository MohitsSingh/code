if (nargin < 4)
    dictSize = 1024;
end
if (nargin < 5)
    suffix = conf.suffix;
end
if (nargin < 6)
    featArgs = {};
end
fullSuffix = [num2str(dictSize) suffix];
dictionaryPath = fullfile(conf.cachedir,['dict_' fullSuffix '.mat']);
if (toSave && exist(dictionaryPath,'file'))
    load(dictionaryPath);
    return;
end


x1 = {};
r = randperm(length(train_ids));
nDescs = dictSize*10;
nImages = 100;
nDescsPerImage = round(nDescs/nImages);
nTotalDescs = 0;
for k = 1:length(r)
    ii = r(k);
    if (ischar(train_ids{ii}))
        I1 = toImage(conf,getImagePath(conf,train_ids{ii}));
    else
        I1 = train_ids{ii};
    end
    [~,descs] = vl_phow(im2single(I1),featArgs{:});
    dSubset = descs(:,vl_colsubset(1:size(descs,2),nDescsPerImage,'Uniform'));
    nTotalDescs = nTotalDescs + size(dSubset,2);
    x1{k} = single(dSubset);
    if (nTotalDescs >= nDescs)
        break;
    end
    fprintf('learnBowDictionary: extracting descriptors : %3.3f%%\n',...
        100*min(nTotalDescs/nDescs,1));
end

x1 = cat(2,x1{:});
[dict] = vl_kmeans(x1, dictSize,'Algorithm','Elkan');
save(dictionaryPath,'dict');