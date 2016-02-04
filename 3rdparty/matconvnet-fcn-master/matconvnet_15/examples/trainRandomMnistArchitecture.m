function net_from_random=trainRandomMnistArchitecture(K)
opts.expDir = fullfile('data','mnist-baseline') ;
opts.train.expDir = opts.expDir ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBnorm = true ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.0001 ;

% opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getMnistImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end
opts.train.gpus=1;
net = cnn_mnist_init('useBnorm', opts.useBnorm) ;


opts.expDir = fullfile('data',['mnist-learn-random-' num2str(K)]);
opts.train.expDir = opts.expDir ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

if exist(opts.imdbPath, 'file')
    load(opts.imdbPath) ;
else
    % create K random networks...
    random_nets = {};
    for u = 1:K
        random_nets{u} = cnn_mnist_init_random('useBnorm', opts.useBnorm) ;
    end
    
    % generate labels.....
    
    imdb_for_random = imdb;
    sets = imdb.images.set;
    images = imdb.images.data(:,:,:,sets==1);
    
    %train/val ratio = 1/6
    sets_new = ones(1,size(images,4));
    sets_new(round(6*end/7):end) = 3;
    
    %labels_new = zeros(K,length(sets_new));
    batches = batchify(length(sets_new),5);
    all_results = {};
    for iNet = 1:length(random_nets)
        iNet
        ret_results = {};
        curNet = random_nets{iNet};
        curNet=vl_simplenn_move(curNet,'gpu');
        curResults = {};
        for iBatch = 1:length(batches)
            %         if (mod(iBatch,5)==0)
            %             iBatch
            %         end
            %         iBatch
            curData = images(:,:,:,batches{iBatch});
            R = vl_simplenn(curNet,gpuArray(curData));
            curResults{iBatch} = gather(squeeze(R(end).x));
        end
        curResults = cat(1,curResults{:});
        all_results{iNet} = curResults;
        curNet=vl_simplenn_move(curNet,'cpu');
    end
    
    new_labels = cat(2,all_results{:});
    
    % save ~/storage/misc/random_supervision_data.mat random_nets new_labels
    
    imdb_for_random.images.data = images;
    L = zeros(1,1,K,size(images,4));
    L(:) = 2*(bsxfun(@minus, new_labels, mean(new_labels)) > 0)'-1;
    imdb_for_random.images.labels = L;
    imdb_for_random.images.set = sets_new;
    % now train a network to be able to predict the random network's outputs.
    
    mkdir(opts.expDir) ;
    save(opts.imdbPath, 'imdb_for_random');
end


net_from_random =  cnn_mnist_init_multi('useBnorm', opts.useBnorm,'K',K) ;
opts.train.errorFunction = 'attributes';
opts.expDir = fullfile('data',['mnist-learn-random-' num2str(K)]);
opts.train.expDir = opts.expDir ;
opts.train.loss = 'logistic';
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
[net_from_random, info] = cnn_train(net_from_random, imdb_for_random, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb_for_random.images.set == 3)) ;