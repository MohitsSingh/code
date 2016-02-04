function classifier = train_region_classifier(all_features,all_labels,curClass,params)
switch params.learning.classifierType
    case 'rand_forest'
        pTrain = {'maxDepth',64,'M',100,'minChild',1};
        myClass = all_labels == curClass;
        classifier =struct('tdata', forestTrain(all_features' ,(myClass+1)', pTrain));
        
    case 'boosting'
        pDebug = 1;
        pBoost = struct('nWeak',round(1024/pDebug),'verbose',1,'pTree',struct('maxDepth',2,'nThreads',8),'discrete',0);
        myClass = all_labels == curClass;
        classifier = adaBoostTrain(all_features(:,~myClass)',all_features(:,myClass)',pBoost);
    case 'svm'
        myClass = all_labels == curClass;
        [ii,jj] = find(isnan(all_features));jj = unique(jj);
        all_features(:,jj) = [];myClass(:,jj) = [];
        if (params.learning.classifierParams.useKerMap)
            all_features = vl_homkermap(all_features,1,'KCHI2');
        end
        [posFeats,negFeats] = splitFeats(all_features,myClass);
        classifier = train_classifier_pegasos(posFeats,negFeats,1,false);
        classifier = struct('tdata',classifier);
end
end