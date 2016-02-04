function res = getScorePerformances(classes,labels,scores,params)
nClasses = length(classes);
for iClass = 1:nClasses
    curClass = classes(iClass);
    res(iClass).class_id = curClass;
    curScores = scores(iClass,:);
    if ~(strcmp(params.task,'regression'))
        curLabels = (labels==curClass)*2-1;
        curScores(isnan(scores)) = -inf;
        [res(iClass).recall, res(iClass).precision, res(iClass).info] = vl_pr(curLabels,curScores);
    end
    res(iClass).curScores = curScores;
    %     curClass
    %     class_names{curClass}
    %     clf; vl_pr(curLabels,curScores); pause
end