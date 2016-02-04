function net = fcnInitializeModel_parametric_prediction_theta2(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16

opts.sourceModelUrl = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = 'data/models/imagenet-vgg-verydeep-16.mat' ;
opts.nClasses=21;
opts = vl_argparse(opts, varargin) ;
% nClasses = opts.nClasses;

% -------------------------------------------------------------------------
%                    Load & download the source model if needed (VGG VD 16)
% -------------------------------------------------------------------------
if ~exist(opts.sourceModelPath)
    fprintf('%s: downloading %s\n', opts.sourceModelUrl) ;
    mkdir(fileparts(opts.sourceModelPath)) ;
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat', opts.sourceModelPath) ;
end
net = load(opts.sourceModelPath) ;
net.layers = net.layers(1:end-1);
net.layers{end}.filters = single(randn(1,1,4096,2)/100);
net.layers{end}.biases = zeros(1,2,'single');
% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

net.layers{end+1} = struct('type', 'pdist', 'name', 'loss','p', 2,'noRoot',true,'epsilon',1e-6 );

% opts.noRoot = false ;
% opts.epsilon = ;
% ) ;


if 0
    figure(100) ; clf ;
    n = numel(net.vars) ;
    for i=1:n
        vl_tightsubplot(n,i) ;
        showRF(net, 'input', net.vars(i).name) ;
        title(sprintf('%s', net.vars(i).name)) ;
        drawnow ;
    end
end




