function normalizers = getNormalizationFactors(conf,train_ids)
% set normalization factors for the different terms, e.g, the general
% classifier and the dpm classifier.

train_ids = vl_colsubset(row(train_ids),100);
allScores = {};
allDPMScores = {};
for k = 1:length(train_ids)
    load(fullfile(conf.classificationDir,strrep(train_ids{k},'.jpg','.mat')));
    allScores{k} = col([regionConfs.score]);
    load(fullfile(conf.dpmDir,strrep(train_ids{k},'.jpg','.mat')));
    curDPMScore = cat(1,modelResults(:).ds);
    allDPMScores{k} = curDPMScore(:,end);
end

allScores = cat(1,allScores{:});
allScores(isnan(allScores)) = [];

allDPMScores = cat(1,allDPMScores{:});
allDPMScores(isnan(allDPMScores)) = [];

% figure,plot(sort(allScores));
normalizers.gmdist_class = gmdistribution.fit(allScores,1);
normalizers.gmdist_dpm = gmdistribution.fit(allDPMScores,1);
% figure,plot(gmdist_dpm.cdf(sort(allDPMScores)))
end