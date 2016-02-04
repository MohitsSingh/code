function [all_results,all_labels]  = getAllResults( opts,imdb,startEpoch,endEpoch,perfSuffix,setToTest )

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% profile off
% profile on

subset = find(imdb.images.set == setToTest); % test
subset = subset(1:1:end);
perfDir = fullfile(opts.expDir,['perf' perfSuffix]);

all_results = {};
all_labels = {};
ensuredir(perfDir);
for epoch = startEpoch:endEpoch
    curPerfPath = fullfile(perfDir,['epoch_' num2str(epoch) '.mat']);
    disp(curPerfPath)
    if exist(curPerfPath,'file')
        load(curPerfPath)
    else
        curBatchResults = {};
        curBatchLabels = {};
        opts.modelPath = fullfile(opts.expDir,sprintf('net-epoch-%d.mat',epoch));
        load(opts.modelPath);net.layers = net.layers(1:end-1);
        getBatch = @(x,y) getSimpleNNBatch(x,y) ;
        opts.batchSize = 1024;
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
    all_results{epoch} = curBatchResults;
    all_labels{epoch} = curBatchLabels;
end

