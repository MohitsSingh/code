function cat_bboxes()
% function cat_bboxes()
%
% Concatenate multiple bounding box files per image into a single file.
%
% Related functions: mat files from run_voting or run_detection
%
% Copyright @ Chunhui Gu April 2009

% Voting or Detection schemes
ntest = 127;
ncategs = 5;
run_dir = 'mat/run1/';

det_bboxes.rect = cell(ncategs,ntest);
det_bboxes.score = cell(ncategs,ntest);

for ii = 1:ntest,
    
    load([run_dir 'detBB' num2str(ii) '.mat']);
    for cc = 1:ncategs,
        det_bboxes.rect{cc,ii} = detBB(cc).rect;
        det_bboxes.score{cc,ii} = detBB(cc).score;
        if isfield(detBB,'vscore'),
            det_bboxes.vscore{cc,ii} = detBB(cc).vscore;
        end;
    end;
end;
if isfield(det_bboxes,'vscore'),
    save([run_dir 'det_bboxes_detection.mat'],'det_bboxes');
else
    save([run_dir 'det_bboxes_voting.mat'],'det_bboxes');
end;

for ii = 1:ntest,
    delete([run_dir 'detBB' num2str(ii) '.mat']);
end;