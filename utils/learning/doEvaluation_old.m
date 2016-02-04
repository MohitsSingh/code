function [recalls,precisions,F_scores,per_class_losses]  = doEvaluation( opts,imdb,startEpoch,endEpoch,perfSuffix )

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% profile off
% profile on

nEpochs = endEpoch-startEpoch+1;
subset = find(imdb.images.set == 2); % test
subset = subset(1:10:end);
perfDir = fullfile(opts.expDir,['perf' perfSuffix]);

resultPath = fullfile(perfDir,'perf_summary.mat');
if exist(resultPath,'file')
    load(resultPath);
    return
end



newVersion = false;

ensuredir(perfDir);
for epoch = startEpoch:endEpoch
    curPerfPath = fullfile(perfDir,['epoch_' num2str(epoch) '.mat']);
    if exist(curPerfPath,'file')
        continue
    end
    curBatchResults = {};
    curBatchLabels = {};
    opts.modelPath = fullfile(opts.expDir,sprintf('net-epoch-%d.mat',epoch));
    load(opts.modelPath);%net.layers = net.layers(1:end-1);
    net = cnn_imagenet_deploy(net);
    meta = net.meta;
    bopts.numThreads = 12;
    bopts.imageSize = meta.normalization.imageSize ;
    bopts.border = meta.normalization.border ;
    bopts.averageImage = meta.normalization.averageImage ;
    bopts.rgbVariance = meta.augmentation.rgbVariance ;
    bopts.transformation = meta.augmentation.transformation ;
    getBatch = @(x,y) getSimpleNNBatch_f(bopts,x,y) ;
    opts.batchSize = 256;
    mode = 'val' ;
    evalMode = 'test' ;
    opts.numSubBatches = 1;
    opts.prefetch = true;
    numGpus=1;
    
    net = vl_simplenn_move(net,'gpu');
    opts.conserveMemory = false;
    opts.backPropDepth = 0;
    opts.sync = false;
    opts.cudnn = false;
    res = [] ;
    mmap = [] ;
    stats = [] ;
    start = tic ;
    
    
    for t=1:opts.batchSize:numel(subset)
        fprintf('%s: epoch %02d: %3d/%3d: ', mode, epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
        batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
        numDone = 0 ;
        error = [] ;
        
        for s=1:opts.numSubBatches
            % get this image batch and prefetch the next
            batchStart = t + (labindex-1) + (s-1) * numlabs ;
            batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
            batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            [im, labels] = getBatch(imdb, batch) ;
            
            if opts.prefetch
                if s==opts.numSubBatches
                    batchStart = t + (labindex-1) + opts.batchSize ;
                    batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
                else
                    batchStart = batchStart + numlabs ;
                end
                nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
                getBatch(imdb, nextBatch) ;
            end
            
            if numGpus >= 1
                im = gpuArray(im) ;
            end
            
            
            
            dzdy = [];
            %     tic
            res = vl_simplenn(net, im, dzdy, res, ...
                'accumulate', s ~= 1, ...
                'mode', evalMode, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'sync', opts.sync, ...
                'cudnn', opts.cudnn) ;
            
            
            curBatchResults{end+1} = squeeze(gather(res(end).x));
            curBatchLabels{end+1} = labels;
            
            numDone = numDone + numel(batch) ;
        end % next sub-batch
        
        % print learning statistics
        time = toc(start) ;
        stats = sum([stats,[0 ; error]],2); % works even when stats=[]
        stats(1) = time ;
        n = t + batchSize - 1 ; % number of images processed overall
        speed = n/time ;
        fprintf('%.1f Hz%s\n', speed) ;
        
        m = n / max(1,numlabs) ; % num images processed on this lab only
        
        fprintf(' [%d/%d]', numDone, batchSize);
        fprintf('\n') ;
    end
    
    curBatchLabels = cat(2,curBatchLabels{:});
    curBatchResults = cat(2,curBatchResults{:});
    
    save(curPerfPath,'curBatchResults','curBatchLabels');
end

%%



all_cms = {};
aucs = zeros(1000,endEpoch);
F_scores = zeros(1000,endEpoch);
recalls = zeros(1000,endEpoch);
precisions = zeros(1000,endEpoch);
class_accuracies = zeros(1000,endEpoch);
per_class_losses = zeros(1000,endEpoch);
%%
for t = startEpoch:endEpoch
    t
    curPerfPath = fullfile(perfDir,['epoch_' num2str(t) '.mat']);
    load(curPerfPath);
    %cm = ...
    %     for tt = 1:1000
    %         [prec,rec,info] = vl_pr(2*(curBatchLabels==tt)-1,curBatchResults(tt,:));
    %         aucs(tt,t) = info.ap;
    %
    %     continue;
    
    [~,IDXpred] = max(curBatchResults,[],1);
    cm = confMatrix( curBatchLabels, IDXpred, 1000 );
    
    curBatchResults = reshape(curBatchResults,1,1,1000,[]);
    loss_opts.loss = 'log';
    [Y,loss_per_class] = vl_nnloss_per_class(curBatchResults,curBatchLabels,[],loss_opts);
    
    % sum the loss per class...
    per_class_loss = zeros(1000,1);
    for ii = 1:length(curBatchLabels)
        jj = curBatchLabels(ii);
        per_class_loss(jj) = per_class_loss(jj)+loss_per_class(ii);
    end
    
    per_class_losses(:,t) = per_class_loss;
    
    %     class_accuracy = zeros(1000,1);
    %     for tt = 1:1000
    %         class_accuracy(tt) = sum(curBatchLabels==tt & IDXpred==tt) + ...
    %             sum(curBatchLabels~=tt & IDXpred~=tt);
    %         class_accuracy(tt) = class_accuracy(tt)/length(curBatchLabels);
    %     end
    %
    %     class_accuracies(:,t) = class_accuracy;
    
    all_cms{t} = cm;
    recall = diag(bsxfun(@rdivide,cm,sum(cm,2)));
    precision = diag(bsxfun(@rdivide,cm,sum(cm,1)));
    F_scores(:,t) = (2*precision.*recall)./(precision+recall);
    recalls(:,t) = recall;
    precisions(:,t) = precision;
end

%%
save(fullfile(perfDir,'perf_summary.mat'),'recalls','precisions','F_scores','per_class_losses','all_cms');


% end

