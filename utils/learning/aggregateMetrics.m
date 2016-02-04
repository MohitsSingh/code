function [ F_scores,cms,perf,precisions,recalls,per_class_losses ] = aggregateMetrics( all_results,all_labels,nClasses)
nEpochs = length(all_results);
F_scores = zeros(nClasses,nEpochs);
precisions = zeros(nClasses,nEpochs);
recalls = zeros(nClasses,nEpochs);
per_class_losses = zeros(nClasses,nEpochs);
cms = {};
for t = 1:length(all_results)
    t
    [per_class_loss,cm1,precision,recall,F_score1] = evaluationMetrics(all_results{t},all_labels{t},nClasses);
    F_scores(:,t) = F_score1;
    precisions(:,t) = precision;
    recalls(:,t) = recall;
    per_class_losses(:,t) = per_class_loss;
    perf(t) = sum(diag(cm1))/sum(cm1(:));
    cms{t} = cm1;
end

end

