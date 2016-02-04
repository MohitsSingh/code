function [feats_train,feats_test] = extractColorFeats(set1,set2,fileName) % conf.suffix = 'rgb';
if (exist(fileName,'file'))
    load (fileName);
    return;
end



feats_train = getColorFeats(set1);
feats_test = getColorFeats(set2);
save(fileName,'feats_train','feats_test');


function feats = getColorFeats(imgs)
feats = {};
for k = 1:length(imgs)
    k
    skinprob = computeSkinProbability(double(imgs{k}));
    normaliseskinprob = normalise(skinprob);
    normaliseskinprob = imresize(normaliseskinprob,[8 8]);
    feats{k} = normaliseskinprob(:);
end
feats = cat(2,feats{:});
