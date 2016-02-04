function [cm,X] = calcPerf2(net,test_images,test_labels,bopts)

opts.conserveMemory = true;
opts.disableDropout = true;

X = {};
for t = 1:length(test_images)
    
%     M = test_images{t};
%     for tt = 1:length(M)
%         M{tt} = single(M{tt});
%     end
%     imo = cnn_imagenet_get_batch(M,bopts);
    
    test_res = vl_simplenn(net,gpuArray(test_images{t}),[],opts);
    %     test_res2 = vl_simplenn(net,gpuArray(test_images(:,:,:,batches{t})),[],opts);
    XX = gather(squeeze(test_res(end).x));
    X{end+1} = XX;
    t/length(test_images)
end

X = cat(2,X{:});
[v,iv] = max(X,[],1);
nClasses = size(X,1);
cm = confMatrix(test_labels,single(iv),nClasses);%,varargin)
cm = bsxfun(@rdivide,cm,sum(cm,2)+eps);
