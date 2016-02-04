function net = fcnInitializeModel_parametric_prediction_theta3(varargin)
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
net.setLayerOutputs('fc8',{'prediction_theta'});
for i = net.getParamIndex(net.layers(end-1).params) ;
    sz = size(net.params(i).value) ;
    sz(end) = 30;
    if length(sz)==4
        sz(1:2) = [1 1];
    end
    net.params(i).value = randn(sz, 'single')/100;    
end
nScales = 8;
% fcnInitializeModel_parametric_prediction_theta



net.addLayer('fc8_scale',dagnn.Conv('size',[1,1,4096,nScales]),'x19','prediction_scale',{'fc8_scalef','fc8_scaleb'});

% net.layers(end-1).block.size = size(...
%     net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

f = net.getParamIndex('fc8_scalef') ;
net.params(f).value = single(randn(1,1,4096,nScales)/100);
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('fc8_scaleb') ;
net.params(f).value = zeros(1,nScales,'single');
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;


% Remove the last loss layer
net.removeLayer('prob') ;

%%

% visualization purposes
net.vars(net.getVarIndex('prediction_scale')).precious = 1 ;
net.vars(net.getVarIndex('prediction_theta')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer for both scale and theta
net.addLayer('objective_scale', ...
    dagnn.Loss('loss', 'softmaxlog'), ...
    {'prediction_scale', 'label_scale'}, 'objective_scale') ;

net.addLayer('objective_theta', ...
    dagnn.Loss('loss', 'softmaxlog'), ...
    {'prediction_theta', 'label_theta'}, 'objective_theta') ;



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




