function [perfs,diags] = test_net_perf(expDir,ppp,imdb,train,val,test,...
    test_params)
% if nargin < 8
%     labels_to_block = [];
% end
% if nargin < 9
%     prefix = 'perfs_ap';
% end

prefix = [test_params.prefix '_' test_params.set];
% gpuDevice(2)
perfs = struct('cm',{},'cm_n',{},'n_epoch',{});
perfDir = fullfile(expDir,prefix);
ensuredir(perfDir);
% ppp = 1:1:nEpochs;
for ipp = 1:length(ppp)
    pp = ppp(ipp);
    modelPath = fullfile(expDir,['net-epoch-' num2str(pp) '.mat']);
    curPerfPath = fullfile(perfDir,[num2str(pp) '.mat']);
    if exist(curPerfPath,'file') && ~test_params.force_check
        load(curPerfPath);
    else % performance file doesn't exis
        if exist(modelPath,'file')
            curPerfPath = fullfile(perfDir,[num2str(pp) '.mat']);
            %             if ~exist(curPerfPath,'file')
            [cm,cm_n] = test_segmentation_accuracy(modelPath,imdb,train,val,test,test_params);
            save(curPerfPath,'cm','cm_n');
        else % neither does the model file, so can't do anything.
            error(sprintf('cannot calculate performance for missing model:%s\n',modelPath));
        end
    end
    perfs(ipp).n_epoch = pp;
    recall = diag(bsxfun(@rdivide,cm,sum(cm,2)));
    precision = diag(bsxfun(@rdivide,cm,sum(cm,1)));
    F_score = (2*precision.*recall)./(precision+recall+eps);
    
    perfs(ipp).cm = cm;
    perfs(ipp).precision = precision;
    perfs(ipp).recall = recall;
    perfs(ipp).F_score = F_score;
    perfs(ipp).cm_n = cm_n;
end
diags = {};
for t = 1:length(perfs)
    diags{t} = perfs(t).F_score;
    %diags{t} = diag(perfs(t).cm_n);
end

diags = cat(2,diags{:})';
