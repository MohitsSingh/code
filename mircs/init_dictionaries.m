function featConf = init_dictionaries(conf)
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[dict_1024,suffix1] = learnBowDictionary(conf,train_ids,true,1024,'',{});
[dict_1024_RGB,suffix2] = learnBowDictionary(conf,train_ids,true,1024,'rgb',{'Color','rgb'});
[dict_1024_OPP,suffix3] = learnBowDictionary(conf,train_ids,true,1024,'opp',{'Color','opponent'});
featConf(1).codebook = dict_1024;
featConf(1).suffix= suffix1;
featConf(1).featArgs = {};
featConf(2).codebook = dict_1024_RGB;
featConf(2).suffix= suffix2;
featConf(2).featArgs = {'Color','rgb'};
featConf(3).codebook = dict_1024_OPP;
featConf(3).suffix= suffix3;
featConf(3).featArgs = {'Color','opponent'};
end