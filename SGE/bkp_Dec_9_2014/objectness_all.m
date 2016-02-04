baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/objectness_s40';
mkdir(outDir);
run_all(baseDir,outDir,'objectness_parallel');