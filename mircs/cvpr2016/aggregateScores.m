function res = aggregateScores(scores,targetInds,n)

res = zeros(size(scores,1),n);
u = unique(targetInds);
for t = 1:length(u)
    m = u(t);
    res(:,m) = max(scores(:,targetInds==m,:),[],2);
end
% res = res(:,u);
end

