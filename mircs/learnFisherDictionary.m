function [dict,fullSuffix] = learnFisherDictionary(conf,train_ids,toSave,dictSize,suffix,featArgs)

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
    dict.means = means;
    dict.covariances = covariances;
    dict.priors = priors;
    return;
end


x1 = {};
r = randperm(length(train_ids));
nDescs = 10^6;
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
fprintf('done extracting descriptors');
x1 = cat(2,x1{:});
[means,covariances,priors] = vl_gmm(x1,dictSize,'Verbose');
dict.means = means;
dict.covariances = covariances;
dict.priors = priors;
% [dict] = vl_kmeans(x1, dictSize,'Algorithm','Elkan');
save(dictionaryPath,'means','covariances','priors');

