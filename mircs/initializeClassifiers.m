function extractors = initializeClassifiers(conf)
% 1. bow classifiers.
for k = 1:length(conf.featConf)
    extractors(k) = BOWFea(conf,conf.featConf(k));
end
extractors(end+1) = HOGClassifier(conf);
end