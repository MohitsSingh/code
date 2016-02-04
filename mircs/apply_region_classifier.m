function decision_values = apply_region_classifier(classifier, features , params)
switch params.learning.classifierType
    case 'rand_forest'
        [hs,probs] = forestApply( features', classifier.tdata);
        decision_values = probs(:,2);
    case 'boosting'
        decision_values = adaBoostApply(features',classifier,[],[],8);
    case 'svm'
        if (params.learning.classifierParams.useKerMap)
            features = vl_homkermap(features,1,'KCHI2');
        end
        decision_values = classifier.w(1:end-1)'*features;     
end
