function classifier = train_classifier_helper_2(classes,features,targetClass,toBalance)
if (nargin < 4)
    toNormalize = false;
end

%[posFeats,negFeats] = splitFeats(features, 2*(classes==targetClass)-1);
[features,labels] = balanceData(features,2*(ismember(classes,targetClass))-1,toBalance);
classifier = train_classifier_pegasos(features,labels);
end