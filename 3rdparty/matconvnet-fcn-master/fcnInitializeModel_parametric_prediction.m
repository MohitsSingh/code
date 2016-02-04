function net = fcnInitializeModel_parametric_prediction(varargin)
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

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------

% Add dropout to the fully-connected layers in the source model
drop1 = struct('name', 'dropout1', 'type', 'dropout', 'rate' , 0.5) ;
drop2 = struct('name', 'dropout2', 'type', 'dropout', 'rate' , 0.5) ;
net.layers = [net.layers(1:33) drop1 net.layers(34:35) drop2 net.layers(36:end)] ;

% Convert the model from SimpleNN to DagNN
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

% Add more padding to the input layer
%net.layers(1).block.pad = 100 ;
net.layers(5).block.pad = [0 1 0 1] ;
net.layers(10).block.pad = [0 1 0 1] ;
net.layers(17).block.pad = [0 1 0 1] ;
net.layers(24).block.pad = [0 1 0 1] ;
net.layers(31).block.pad = [0 1 0 1] ;
net.layers(32).block.pad = [3 3 3 3] ;
% ^-- we could do [2 3 2 3] but that would not use CuDNN

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
    if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1}) ;
        bias = net.getParamIndex(net.layers(i).params{2}) ;
        net.params(bias).learningRate = 2 * net.params(filt).learningRate ;
    end
end

%%
% create B basis functions
basisFilters = {};
%filt_size = 33;
filt_size = 64;
T = zeros(filt_size,'single');
T(:,end/3:2*end/3) = 1;
thetas = 0:20:160;
nBase = length(thetas);
for iR = 1:nBase
    r = thetas(iR);
    V = imrotate(T,r,'bicubic','crop');
    %     V = V/sum(V(:));
    %     V = zeros(size(V),'single');
    %     V(ceil(end/2),ceil(end/2)) = 1;
    basisFilters{end+1} = V;
end
% x2(basisFilters);

basisFilters = cat(3,basisFilters{:});
% 
%%
% Modify the last fully-connected layer to have B output coefficients, for the basis functions
% Initialize the new filters to zero (MODIFIED TO GAUSSIAN NOISE,AMIR)
for i = net.getParamIndex(net.layers(end-1).params) ;
    sz = size(net.params(i).value) ;
    sz(end) = nBase;
    if length(sz)==4
        sz(1:2) = [7 7];
    end
    net.params(i).value = randn(sz, 'single')/100;
    
end
net.layers(end-1).block.size = size(...
    net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
% net.removeLayer('prob') ;
net.setLayerOutputs('fc8', {'x38'}) ;
% 
% net.addLayer('x38_relu', ...
%     dagnn.ReLU('useShortCircuit',0),...
%     'x38_0', 'x38',{}) ;

% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------
N = nBase;
nGroups=N;
filters = single(bilinear_u(224, nGroups, N)) ;

% N = nBase;
% nGroups=N;
for f = 1:N
    %     filters(:,:,:,1)
    %     filters(:,:,f,1) = 1-imResample(basisFilters(:,:,f),[64 64],'bilinear');
    filters(:,:,:,f) = imResample(basisFilters(:,:,f),[224 224],'bilinear');
    %
end


net.addLayer('deconv32', ...
    dagnn.ConvTranspose(...
    'size', size(filters), ...
    'upsample', 224, ...
    'crop', 0*[16 16 16 16], ...
    'numGroups', nGroups, ...
    'hasBias', true), ...
    'x38', 'pre_prediction', {'deconvf','deconvb'});

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('deconvb') ;
net.params(f).value = zeros(1,size(filters,4),'single');
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;


%%

% -------------------------------------------------------------------------
% Combination of basis functions
% -------------------------------------------------------------------------

N = opts.nClasses+1;
% filters = cat(4,basisFilters,1-basisFilters);
filters = ones(1,1,9,2,'single');
filters(:,:,:,2) = -filters(:,:,:,1);
filters=filters/18;
% net.addLayer('conv_basis', ...
%   dagnn.Conv('size', size(filters), 'pad', [16 16 16 16]),...
%   'pre_prediction', 'prediction', {'conv_basis_f','conv_basis_b'});

net.addLayer('conv_basis', ...
    dagnn.Conv('size', size(filters), 'pad', 0),...
    'pre_prediction', 'prediction_0', {'conv_basis_f','conv_basis_b'});


net.addLayer('prediction_relu', ...
    dagnn.ReLU('useShortCircuit',0),...
    'prediction_0', 'prediction',{}) ;


biases = zeros(1,2,'single');
f = net.getParamIndex('conv_basis_f') ;
net.params(f).value = filters;
f = net.getParamIndex('conv_basis_b') ;
net.params(f).value = biases;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
    SegmentationLoss('loss', 'softmaxlog'), ...
    {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
    SegmentationAccuracy(), ...
    {'prediction', 'label'}, 'accuracy') ;

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




