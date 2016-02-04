function res = seg_new_parallel(conf,I,reqInfo)

if (nargin == 0)
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));
    cd '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';
    install;
    res = [];return;
end

I = imread(I);
[candidates, ucm2] = im2mcg(I,'accurate');
res.cadidates = candidates;
res.ucm2 = single(ucm2);
