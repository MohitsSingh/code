function classifier_data = train_stage_classifier(conf,fra_db,params,prevClassifierData)

classifier_data = struct('trainInds',{},'valInds',{},'classifiers',{},'val_results',{});
% classifier_data(1).trainInds = trainSet;
% classifier_data.valInds = valSet;
max_neg_to_keep = params.learning.maxNegsToKeep;

if (nargin < 4)
    [curFeats,curLabels,ovps] = collect_feature_subset(conf,fra_db,params,max_neg_to_keep);
else
    [prevFeats,prevLabels,ovps] = collect_feature_subset(conf,fra_db,prevClassifierData.params,max_neg_to_keep);
    % validation set...
    %prevClassifierScores = applyToImageSet(fra_db(valSet),prevClassifierData.classifiers,params);
    [curFeats,curLabels] = collect_feature_subset(conf,fra_db,max_neg_to_keep);
    classifiers = {};
    
    for iClass = 1:length(classes)
        classifiers{iClass} = train_region_classifier(curFeats,curLabels,classes(iClass),params);
    end
    classifiers = [classifiers{:}];
end
classifier_data.classifiers = classifiers;
end