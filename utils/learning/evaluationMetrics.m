function [per_class_loss,cm,precision,recall,F_score] = evaluationMetrics(results,labels,N)
[~,IDXpred] = max(results,[],1);
% cm1 = confMatrix( labels, IDXpred, N );

cm = accumarray([labels(:), IDXpred(:)],1,[N N]);

results = reshape(results,1,1,N,[]);
% loss_opts.loss = 'log';
[Y,loss_per_class] = vl_nnloss_per_class(results,labels,[],'loss','softmaxlog');
% sum the loss per class...
per_class_loss = zeros(N,1);
for ii = 1:length(labels)
    jj = labels(ii);
    per_class_loss(jj) = per_class_loss(jj)+loss_per_class(ii);
end
% per_class_losses = per_class_loss;
recall = diag(bsxfun(@rdivide,cm,sum(cm,2)));
precision = diag(bsxfun(@rdivide,cm,sum(cm,1)));
F_score = (2*precision.*recall)./(precision+recall);
