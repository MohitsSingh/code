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
filt_size = 224;
T = zeros(filt_size,'single');
T(:,end/3:2*end/3) = 1;
thetas = 0:20:160;
nBase = length(thetas);
for iR = 1:nBase
    r = thetas(iR);
    V = imrotate(T,r,'bicubic','crop');
    V = V/sum(V(:));
%     V = zeros(size(V),'single');
%     V(ceil(end/2),ceil(end/2)) = 1;
    basisFilters{end+1} = V;
end
% x2(basisFilters);

basisFilters = cat(3,basisFilters{:});

%%
% Modify the last fully-connected layer to have B output coefficients, for the basis functions
% Initialize the new filters to zero
for i = net.getParamIndex(net.layers(end-1).params) ;
  sz = size(net.params(i).value) ;
  sz(end) = nBase;
  net.params(i).value = zeros(sz, 'single') ;
end
net.layers(end-1).block.size = size(...
  net.params(net.getParamIndex(net.layers(end-1).params{1})).value) ;

% Remove the last loss layer
net.removeLayer('prob') ;
net.setLayerOutputs('fc8', {'x38'}) ;

% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------
N = nBase; 
nGroups=N;
filters = single(bilinear_u(64, nGroups, 2)) ;
for f = 1:N
    filters(:,:,f,1) = 1-imResample(basisFilters(:,:,f),[64 64],'bilinear');
    filters(:,:,f,2) = imResample(basisFilters(:,:,f),[64 64],'bilinear');
    
end
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 32, ...
  'crop', [16 16 16 16], ...
  'numGroups', nGroups, ...
  'hasBias', false), ...
  'x38', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

%%


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




