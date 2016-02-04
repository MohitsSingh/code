function res = classificationHelper(all_feats,train_set,valids,iClass,all_class_ids,excludeClass,useRandomForest)
if nargin < 6
    excludeClass = -1;
end
if (nargin < 7)
    useRandomForest=1;
end

valids = valids & all_class_ids~=excludeClass;
sel_class = all_class_ids==iClass;
sel_pos = valids & train_set & sel_class;
sel_neg = valids & train_set & ~sel_class;
features_pos = cat(2,all_feats{sel_pos});
features_pos = features_pos';
features_neg = cat(2,all_feats{sel_neg});
features_neg = features_neg';
sel_test = ~train_set & valids;
feats_test = cat(2,all_feats{sel_test})';
if (useRandomForest)
    data = [features_pos;features_neg];
    hs = [ones(1,size(features_pos,1)) 2*ones(1,size(features_neg,1))];
%     pTrain={'maxDepth',15,'F1',16,'M',10,'minChild',1}; % 3 ,5  % M==20
    pTrain={'maxDepth',5,'M',1,'minChild',3}; % 3 ,5  % M==20
    forest = forestTrain(data , hs, pTrain);
    [r,probs] = forestApply( feats_test, forest);
    res = probs(:,1);
else
    normalizeFeats = false;
    classifier = train_classifier_pegasos(double(features_pos'),double(features_neg'),1,normalizeFeats)
    if (normalizeFeats)
        feats_test = bsxfun(@rdivide,feats_test,sum(feats_test.^2,2).^.5);
    end
    res = feats_test*classifier.w(1:end-1);
    %         res(n).classifier.w =           classifier.w;
    %         res(n).classifier.optAvgPrec =  classifier.optAvgPrec;
    %         res(n).classifier.optLambda =   classifier.optLambda;
    %         res(n).feat_name = feat_names{iFeatType};
    %         res(n).feat_id = curFeatType;
    %         % calculate results for each image...
    %         sel_feat_type_test = sel_feat_type & ~train_set;
    %         features_test = cat(2,all_feats(sel_feat_type_test).feat);
    %         test_scores = res(n).classifier.w(1:end-1)'* features_test +res(n).classifier.w(end);
    %         test_inds = all_image_inds(sel_feat_type_test);
end