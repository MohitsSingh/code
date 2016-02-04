function extractors = initializeFeatureExtractors(conf)
% 1. bow classifiers.
extractors = {};
% for k = 1


% ffe = VladFeatureExtractor(conf,conf.featConf(4));
ffe = FisherFeatureExtractor(conf,conf.featConf(4));
bfe = BOWFeatureExtractor(conf,conf.featConf(1)); % use all feature types.
ffe.doPostProcess = false;
ffe.useRectangularWindows = false;
ffe.resizeWindows = [];
extractors{1} = ffe;
% for k = 1:length(conf.featConf)
%     curExtractor = BOWFeatureExtractor(conf,conf.featConf(k));
%     curExtractor.useRectangularWindows = false;
%     curExtractor.resizeWindows = [];
%     extractors{k} = curExtractor;
% end
% end

% shape feature extractors...
% extractors{end+1} = HOGFeatureExtractor(conf);
% extractors{end+1} = MaskHOGFeatureExtractor(conf);
% extractors{end+1} = MaskShapeFeatureExtractor(conf);
% extractors{end+1} = RegionPropFeatureExtractor(conf);

% for k = 1:length(extractors)
%     extractors{k}.doPostProcess = true;
% end

end