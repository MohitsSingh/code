% learn some objects using the sun database.
baseDir = '/home/amirro/storage/root4sun';
cd(baseDir);
addpath(genpath('.'));%compile;

cd voc-release5
pascal('bottle',3)
pascal('mug',3)
pascal('can',3)
pascal('glass',3)
pascal('hand');