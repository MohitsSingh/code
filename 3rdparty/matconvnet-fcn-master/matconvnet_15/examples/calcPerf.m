function [cm,X] = calcPerf(net,test_images,test_labels,subsample)


if nargin < 4
    subsample=1;
end

X = {};
test_images = test_images(:,:,:,1:subsample:end);
test_labels = test_labels(1:subsample:end);
batches = batchify(size(test_images,4),20);
opts.conserveMemory = true;
opts.disableDropout = true;
for t = 1:length(batches)
    
    test_res = vl_simplenn(net,gpuArray(test_images(:,:,:,batches{t})),[],opts);
    %     test_res2 = vl_simplenn(net,gpuArray(test_images(:,:,:,batches{t})),[],opts);
    XX = gather(squeeze(test_res(end).x));
    X{end+1} = XX;
    t/length(batches)
end

X = cat(2,X{:});
[v,iv] = max(X,[],1);
nClasses = size(X,1);
cm = confMatrix(test_labels,single(iv),nClasses);%,varargin)
cm = bsxfun(@rdivide,cm,sum(cm,2)+eps);
