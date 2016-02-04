function [dict] = learnDictionary(VOCopts,trainIDs)
%LEARNDICTIONARY Summary of this function goes here
%   Detailed explanation goes here
dictionaryPath = 'data/dictionary.mat';
if (exist(dictionaryPath,'file'))
    load(dictionaryPath);
else
    phowOpts = {'Sizes', 4, 'Step', 30,'Color','PHOW-color'};
    for ii = 1:30:length(trainIDs)
        ii
        im = im2single(readImage(VOCopts,trainIDs{ii}));
        [drop, descrs{ii}] = vl_phow(im, phowOpts{:});
    end
    numWords = 400;
    
    descrs = vl_colsubset(cat(2, descrs{:}), 10e3);
    dict = vl_kmeans(single(descrs),numWords,'Algorithm','Elkan');
    save(dictionaryPath,'dict');
end

