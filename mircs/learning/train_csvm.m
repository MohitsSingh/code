function classifier = train_csvm(classes,features,targetClass,toBalance,clusters)

D = l2(features',clusters);
[~,IC] = min(D,[],2);

classifiers = {};
for u = 1:size(clusters,1)
    %[posFeats,negFeats] = splitFeats(features, 2*(classes==targetClass)-1);
    f = find(IC==u);
    [features_,labels_] = balanceData(features(:,f),2*(classes(f)==targetClass)-1,toBalance);
    classifiers{u} = train_classifier_pegasos(features_,labels_);
end
classifier = cat(2,classifiers{:});
end