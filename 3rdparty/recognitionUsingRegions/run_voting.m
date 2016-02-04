function run_voting(testId)
% function run_voting(testId)
%
% This function works with torun_voting (on the cluster) which is
% equivalent to test_voting.
%
% Related function: main_voting
%
% Copyright @ Chunhui Gu, April 2009

ncategs = 5;
thres = 0.1; % threshold for voting score
run_dir = 'mat/run1/';

if ~exist([run_dir 'detBB' num2str(testId) '.mat'],'file'),
    load([run_dir 'filename.mat']);
    load([run_dir 'training_data_wpos_weights.mat']);
    disp('Done loading data!');

    filename = test_name{testId};
    for categId = 1:ncategs,
        detBB(categId) = main_voting(filename,categId,thres,images,base_dir);
    end;
    save([run_dir 'detBB' num2str(testId) '.mat'],'detBB');
end;