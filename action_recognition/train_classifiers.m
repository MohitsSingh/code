function [ classifier_data ] = train_classifiers(conf,imdb)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (nargin < 7)
    max_neg_to_keep = inf;
end
train_set = imdb.train_set;
classifier_data(1).trainInds = trainSet;
classifier_data.valInds = valSet;
%
[curFeats,curLabels] = collect_feature_subset(imdb(trainSet),params,max_neg_to_keep);
classifiers = {};

for iClass = 1:length(classes)
    classifiers{iClass} = train_region_classifier(curFeats,curLabels,classes(iClass),params);
end
classifiers = [classifiers{:}];
classifier_data.classifiers = classifiers;

end

end

