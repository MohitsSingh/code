% Parallel process demo
% (Should be run from a cluster node)

disp('A simple example of run_parallel');

% filenames
% files = strcat('~/temp_', {'A' 'B' 'C'}, '.mat');

% parallel code - save mat files with the following variables:
%   x = 10 | 20 | 30
%   s = 'hello'
% note: this runs synchronously


%code = 'save(filename, ''x'', ''s'')';
code = 'cd /home/amirro/code/SGE; facial_landmarks_script(nimage)';
% code = 'cd /home/amirro/code/SGE; do_stuff(nimage)';
load ('/home/amirro/code/mircs/m_test.mat');
d = 1:length(m_test);
images = d(randperm(length(d)));
% images =images(1:5);

run_parallel(code, 'nimage',mat2cell(images,1,ones(1,length(images))),'-cluster', 'mcluster01');

% run_parallel(code, 'filename', files, 'x', {10 20 30}, 's', 'hello', '-cluster', 'mcluster01');

% display file contents and delete files
% for i = 1:length(files)
%     load(files{i});
%     files{i}, x, s,
%     delete(files{i});
% end