function M = occPatternsToMat(patterns)
if (isempty(patterns))
    M = [];
    return;
end
patterns = rmfield(patterns,'angular_coverage_min_f');
patterns = rmfield(patterns,'angular_coverage_max_f');
patterns = rmfield(patterns,'dist_coverage');
fieldNames = fieldnames(patterns);
M = {};
for t = 1:length(fieldNames)
    eval(['a=[patterns.' fieldNames{t} '];']);
    M{t} = a;
end
M = cat(1,M{:});