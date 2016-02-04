function featConf = init_features(conf,dictSize)
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');

% create a set of feature extractors...
if (nargin < 2)
    dictSize = 1024;
end

[dict_fisher,suffix_fisher] = learnFisherDictionary(conf,train_ids,true,256,'_fisher',{});
% [dict_fisher,suffix_fisher] = learnFisherDictionary(conf,train_ids,true,64,'_fisher_64',{});
[dict_fisher_64,suffix_fisher_64] = learnFisherDictionary(conf,train_ids,true,64,'_fisher',{});

[dict,suffix1] = learnBowDictionary(conf,train_ids,true,dictSize,'',{});
[dict,suffix1] = learnBowDictionary(conf,train_ids,true,dictSize,'',{});

[dict_RGB,suffix2] = learnBowDictionary(conf,train_ids,true,dictSize,'rgb',{'Color','rgb'});
[dict_OPP,suffix3] = learnBowDictionary(conf,train_ids,true,dictSize,'opp',{'Color','opponent'});
featConf(1).bowmodel.vocab = dict;
featConf(1).suffix = suffix1;
featConf(1).featArgs = {'Step',2};
featConf(2).bowmodel.vocab = dict_RGB;[dict_fisher,suffix_fisher] = learnFisherDictionary(conf,train_ids,true,256,'_fisher',{});
featConf(2).suffix = suffix2;
featConf(2).featArgs = {'Step',2,'Color','rgb'};
featConf(3).bowmodel.vocab = dict_OPP;
featConf(3).suffix = suffix3;
featConf(3).featArgs = {'Step',2,'Color','opponent'};
featConf(4).bowmodel.vocab = dict_fisher_64;
featConf(4).suffix = suffix_fisher_64;
% featConf(4).bowmodel.vocab = dict_fisher;
% featConf(4).suffix = suffix_fisher;

featConf(4).featArgs = {'Step',2};

featConf(5).bowmodel.vocab = dict_fisher;
featConf(5).suffix = '_vlad';
featConf(5).featArgs = {'Step',2};


for k = 1:length(featConf)
    featConf(k).bowmodel.numSpatialX = conf.bow.tiling;
    featConf(k).bowmodel.numSpatialY = conf.bow.tiling;   
end

 featConf(4).bowmodel.numSpatialY = [3 1];
 featConf(4).bowmodel.numSpatialX = [1 1];