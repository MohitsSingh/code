function [imageInds,scores] = summarizeResults(res)
scores = {};
imageInds = res.valInds;
for t = 1:length(res.valInds)
    scores{t} = max(res.val_results(t).decision_values,[],2);
end
scores = cat(2,scores{:});
end