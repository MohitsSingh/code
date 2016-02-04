function [regionConfs,modelResults] = normalizeScores(normalizers,regionConfs,modelResults)
for k = 1:length(regionConfs)
    regionConfs(k).score = normalizers.gmdist_class.cdf(regionConfs(k).score);
end
for k = 1:length(modelResults)
    modelResults(k).ds(:,end) = normalizers.gmdist_dpm.cdf(modelResults(k).ds(:,end));
    
end
end