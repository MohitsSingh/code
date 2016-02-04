function [scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar,predVar)
im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;
% Some networks requires the image to be a multiple of 32 pixels
if imageNeedsToBeMultiple
    sz = [size(im,1), size(im,2)] ;
    sz_ = round(sz / 32)*32 ;
    im_ = imresize(im, sz_) ;
else
    im_ = im ;
end
net.eval({inputVar, gpuArray(im_)}) ;
scores = gather(net.vars(net.getVarIndex(predVar)).value);
[~,pred_] = max(scores,[],3) ;
if imageNeedsToBeMultiple
    pred = imResample(pred_, sz, 'nearest');
else
    pred = pred_ ;
end

scores = imResample(scores,size2(rgb));

