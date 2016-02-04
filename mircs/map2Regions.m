function [scoreMaps,regionScores] = map2Regions(I,regions,modelResults)
scoreMaps = cell(1,length(modelResults));
regionScores = zeros(length(modelResults),length(regions));
for k = 1:length(scoreMaps)
    scoreMaps{k} = computeHeatMap(I,modelResults(k).ds(:,[1:4 6]));
    scoreMaps{k} = scoreMaps{k}/max(scoreMaps{k}(:));
    regionScores(k,:) = cellfun(@(x) mean(scoreMaps{k}(x)), regions);
end
end