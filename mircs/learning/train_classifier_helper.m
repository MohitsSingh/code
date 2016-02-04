function classifier = train_classifier_helper(fra_db,all_feats,targetClass,toNormalize,valids,useTestExamples)
if (nargin < 4)
    toNormalize = false;
end

if (nargin < 5)
    valids = true(size(fra_db));
end

if (nargin < 6)
    useTestExamples = false;
end
if (useTestExamples)
    train_set = true(size(fra_db)) & valids;
else
    train_set = [fra_db.isTrain] & valids;
end
class_ids = [fra_db.classID];
[posFeats,negFeats] = splitFeats(all_feats(:,train_set),...
    class_ids(train_set)==targetClass);
classifier = train_classifier_pegasos(posFeats,negFeats,0,toNormalize);
end