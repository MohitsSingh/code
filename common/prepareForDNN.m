function imo = prepareForDNN(imgs,net,prepareSimple)
if (nargin < 3)
    prepareSimple = false;
end
if (prepareSimple)
    net.normalization.averageImage = [];
end


for z = 1:size(imgs)
    if (length(size(imgs{z}))==2)
        imgs{z} = cat(3,imgs{z},imgs{z},imgs{z});
    end
end

F = cellfun(@(x) single(im2uint8(x)),imgs,'UniformOutput',false);
% imo = cnn_imagenet_get_batch(F, 'averageImage',net.normalization.averageImage,...
%     'border',net.normalization.border,'keepAspect',net.normalization.keepAspect,...
%     'numThreads', 1, ...
%     'prefetch', false,...
%     'imageSize',net.normalization.imageSize);



% opts.imageSize = [227, 227] ;
% opts.border = [29, 29] ;
% opts.keepAspect = true ;
% opts.numAugments = 1 ;
% opts.transformation = 'none' ;
% opts.averageImage = [] ;
% opts.rgbVariance = zeros(0,3,'single') ;
% opts.interpolation = 'bilinear' ;
% opts.numThreads = 1 ;
% opts.prefetch = false ;
% opts = vl_argparse(opts, varargin);
% 
imo = cnn_imagenet_get_batch(F, 'averageImage',net.normalization.averageImage,...
    'border',net.normalization.border,'keepAspect',true,...
    'numThreads', 1, ...
    'prefetch', false,...
    'imageSize',net.normalization.imageSize,'transformation','f5');