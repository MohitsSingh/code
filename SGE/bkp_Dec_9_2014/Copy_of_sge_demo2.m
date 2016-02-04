% Parallel process demo
% (Should be run from a cluster node)

disp('A simple example of run_parallel');

% filenames
files = strcat('~/temp_', {'A' 'B' 'C'}, '.mat');

% parallel code - save mat files with the following variables:
%   x = 10 | 20 | 30
%   s = 'hello'
% note: this runs synchronously

code = 'cd /home/amirro/code/pose; run_gpb(k);';

run_parallel(code, 'k', num2cell(1:30), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(31:60), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(61:90), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(91:120), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(121:150), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(151:180), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(181:210), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(211:240), '-cluster', 'mcluster01');
run_parallel(code, 'k', num2cell(241:270), '-cluster', 'mcluster01');

%run_parallel(code, 'filename', files, 'x', {10 20 30}, 's', 'hello', '-cluster', 'mcluster01');

% % display file contents and delete files
% for i = 1:length(files)
%     load(files{i});
%     files{i}, x, s,
%     delete(files{i});
% end