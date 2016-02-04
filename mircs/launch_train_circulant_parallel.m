%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function clusters = launch_train_circulant_parallel(conf,clusters,naturalSet,job_suffix)
%% Part 1: General Usage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
extraInfo.path = path;
extraInfo.naturalSet = naturalSet;
clusters = run_and_collect_results(clusters,'train_circulant_parallel',false,extraInfo,job_suffix,50);
cd(curDir);

