% this function runs the entire detection scheme (voting and verification)
% on all test images and evaluate detection performance
%
% Related function: main_voting, main_verify
%
% Copyright @ Chunhui Gu, April 2009

run_dir = 'mat/run1/';

load([run_dir 'filename.mat']);
load([run_dir 'training_data_wpos_weights.mat']);
disp('Done loading data!');

ncategs = 5;
ntest = 127;
thres = 0.1; % threshold for voting score

det_bboxes.rect = cell(ncategs,ntest);
det_bboxes.score = cell(ncategs,ntest);

for testId = 1:ntest,
    filename = test_name{testId};
    for categId = 1:ncategs,
        detBB = main_voting(filename,categId,thres,images,base_dir);
        detBB = main_verify(filename,categId,images,base_dir,detBB);
        det_bboxes.rect{categId,testId} = detBB.rect;
        det_bboxes.score{categId,testId} = detBB.score .* detBB.vscore;
    end;
end;

load([run_dir 'gt_bound_mask.mat']);
main_eval(det_bboxes, test_bound, test_class);