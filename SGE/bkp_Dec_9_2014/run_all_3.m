function run_all_3(inputs,outDir,fun,justTesting,cluster_name)

if (nargin < 6 || isempty(cluster_name))
    cluster_name = 'mcluster03';
end
if (~isfield(inputs,'name'))
    error('input must contain field ''name''');
end
ranges = {};
nSplits = 100; % TODO - was 100
inputs = inputs(randperm(length(inputs)));
for k = 0:nSplits-1
    ranges{k+1} = inputs(mod(1:length(inputs),nSplits)==k);
end
ranges = ranges(cellfun(@(x) ~isempty(x),ranges)); % remove empty input arrays
mkdir(outDir);


if (justTesting)
    parallel_helper_2(fun,ranges{1},outDir);
    return
end

% create parallel code
code = ['cd /home/amirro/code/SGE; ' 'parallel_helper_2(fun,input,outDir)'];

fprintf('Parallel code:\n%s\n', code);
% run parallel code
fprintf('Running in parallel:\n');
echo on;

run_parallel(code, 'fun',fun,'input',ranges,'indRange',ranges,'outDir',outDir,...
    '-cluster', cluster_name);
echo off;
fprintf('Done\n');

end