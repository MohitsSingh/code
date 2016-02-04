% Parallel process demo
% (Should be run from a cluster node)

fprintft('Testing parallel process library');
fprintft('We will save some data to files');

% create some data
filenames = strcat('~/sge_test_', {'1'; '2'; '3'; '4'; '5'}, '.mat');
x = num2cell(10 .* (1:5));
y = 'This is a test string';
fprintft('Data:');
disp('files =');
disp(filenames);
disp('x =');
disp(x);
disp('y =');
disp(y);

% create parallel code
code = 'save(filename, ''x'', ''y'')';
fprintft('Parallel code:\n%s', code);

% run parallel code
fprintft('Running in parallel');
run_parallel(code, 'filename', filenames, 'x', x, 'y', y);
fprintft('Done');
