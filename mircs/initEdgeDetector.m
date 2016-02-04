function model = initEdgeDetector()
p = pwd;
addpath('/home/amirro/code/3rdparty/structured_edge_detection/');
%%
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelFinal';       % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~30m/15Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.nTreesEval=model.opts.nTrees;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=1;
cd(p);
end
