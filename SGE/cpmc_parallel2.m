function allres = cpmc_parallel(imagePaths,inds,jobID,outDir,extraInfo,job_suffix)


cd /home/amirro/code/3rdparty/cpmc_release1/;
addpath('./code/');
addpath('./external_code/');
addpath('./external_code/paraFmex/');
addpath('./external_code/imrender/vgg/');
addpath('./external_code/immerge/');
addpath('./external_code/color_sift/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/mex/mexa64/');
addpath('./external_code/vlfeats/toolbox/mex/mexglx/');
addpath('./external_code/globalPb/lib/');
addpath('./external_code/mpi-chi2-v1_5/');
addpath('./code/');
addpath('./external_code/');
addpath('./external_code/paraFmex/');
addpath('./external_code/imrender/vgg/');
addpath('./external_code/immerge/');
addpath('./external_code/color_sift/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/mex/mexa64/');
addpath('./external_code/vlfeats/toolbox/mex/mexglx/');
addpath('./external_code/globalPb/lib/');
addpath('./external_code/mpi-chi2-v1_5/');

% create multiple threads (set how many you have)

amir_keep =false;
if (amir_keep)
    
    N_THREADS = 8;
    if(matlabpool('size')~=N_THREADS)
        matlabpool('open', N_THREADS);
    end
end

%exp_dir = './data/';
exp_dir = '/net/mraid11/export/data/amirro/cpmc_data/';
img_name = '2007_009084'; % dogs, motorbike, chairs, people

[masks, scores] = cpmc(exp_dir, img_name);
res.masks = masks;
res.scores = scores;