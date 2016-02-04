function m = segmentnessMeasure(regions,ucm)
% m = segmentnessMeasure(regions,ucm)
% Compute UCM statistics along region boundary.
m = zeros(size(regions));
for k = 1:length(regions)
    r = imdilate(bwperim(regions{k}),ones(3));
    r = r & ucm > 0;
    m(k) = mean(ucm(r));
end
end