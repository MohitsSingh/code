function res = train_classifiers_folds( train_data,train_labels,train_ovps,train_params,inds_folds)
% RES = TRAIN_CLASSIFIERS_FOLDS(train_data,train_labels,train_params,inds_folds)
% train a classifier while holding out one fold at a time, where data folds are
% specified in inds_folds. This is a thin wrapper for train_classifiers.

folds = unique(inds_folds);
res = {};
for iFold = 1:length(folds)
    curFold = inds_folds~=folds(iFold);
    r = train_classifiers( train_data(:,curFold),train_labels(curFold),train_ovps(curFold),train_params);
    r.fold = find(curFold);
    res{iFold} = r;
end
res = [res{:}];
end