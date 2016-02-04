% Parallel process demo
% (Should be run from a cluster node)

fprintf('Testing parallel process library\n');
fprintf('We will save some data to files\n');

% create some data
filenames = strcat('~/sge_test_', {'1'; '2'; '3'; '4'; '5'}, '.mat');
x = num2cell(10 .* (1:5));
y = 'This is a test string';
fprintf('Data:\n');
disp('files =');
disp(filenames);
disp('x =');
disp(x);
disp('y =');
disp(y);

% create parallel code
code = 'save(filename, ''x'', ''y'')';
fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;
run_parallel(code, 'filename', filenames, 'x', x, 'y', y, '-cluster', 'mcluster01');
echo off;
fprintf('Done\n');
