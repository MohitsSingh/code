function net = fcnInitializeModel_parametric_prediction_theta(varargin)
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
% net.layers = 
% % -------------------------------------------------------------------------
% %                                  Edit the model to create the FCN version
% % -------------------------------------------------------------------------
% 
% % Add dropout to the fully-connected layers in the source model
% 
% % Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
% 
% % Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
    if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1}) ;
        bias = net.getParamIndex(net.layers(i).params{2}) ;
        net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
    end
end


%%
% Modify the last fully-connected layer to have B output coefficients, for the basis functions
% Initialize the new filters to zero (MODIFIED TO GAUSSIAN NOISE,AMIR)

% 30 rotations, 8 scales
% net.setLayerOutputs('fc8',{'prediction_theta'});
for i = net.getParamIndex(net.layers(end-1).params) ;
    sz = size(net.params(i).value) ;
    sz(end) = imdb.nClasses;
    if length(sz)==4
        sz(1:2) = [1 1];
    end
    net.params(i).value = randn(sz, 'single')/100;    
end

%(obj, name, block, inputs, outputs, params)
% 
% net.addLayer('skip3', ...
%      dagnn.Conv('size', [1 1 256 N]), ...
%      'x17', 'x43', {'skip3f','skip3b'});


% net.addLayer('fc8_scale',dagnn.Conv('size',1,1,4096,8),

net.layers(end-1).block.size = size(...
    net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
% net.removeLayer('prob') ;
net.setLayerOutputs('fc8', {'prediction'}) ;

%%

% -------------------------------------------------------------------------
% Combination of basis functions
% -------------------------------------------------------------------------

% N = opts.nClasses+1;
% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
    dagnn.Loss('loss', 'softmaxlog'), ...
    {'prediction', 'label'}, 'objective') ;

% net.addLayer('objective',dagnn.Loss

%net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
%    {'prediction','label'}, 'top1err') ;


% net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
%     'opts', {'topK',5}), ...
%     {'prediction','label'}, 'top5err') ;
% 
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




