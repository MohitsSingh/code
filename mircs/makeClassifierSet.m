function classifier_data = makeClassifierSet(fra_db,trainSet,valSet,params,classes,classifiers,max_neg_to_keep)
if (nargin < 7)
    max_neg_to_keep = inf;
end
classifier_data = struct('trainInds',{},'valInds',{},'classifiers',{},'val_results',{});
classifier_data(1).trainInds = trainSet;
classifier_data.valInds = valSet;
if (~exist('classifiers','var') || isempty(classifiers))
    %
    [curFeats,curLabels] = collect_feature_subset(fra_db(trainSet),params,max_neg_to_keep);
    classifiers = {};    
    for iClass = 1:length(classes)
        classifiers{iClass} = train_region_classifier(curFeats,curLabels,classes(iClass),params);
    end
    classifiers = [classifiers{:}];
end
classifier_data.classifiers = classifiers;
classifier_data.val_results = applyToImageSet(fra_db(valSet),classifiers,params);
end