% learn some objects using the pascal dataset - using the "no hard negative
% mining" approach
baseDir = '/home/amirro/storage/root4sun';
cd(baseDir);
addpath(genpath('.'));%compile;

cd voc-release5
pascal('bottle',3)
pascal('mug',3)
pascal('can',3)
pascal('glass',3)