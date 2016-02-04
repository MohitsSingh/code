function net = cnn_for_probs(varargin)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST
opts.useBnorm = true ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/10 ;
net.layers = {};

% net.layers{end+1} = struct('type', 'conv', ...
%    'weights', {{f*randn(3,3,8,8, 'single'), zeros(1, 8, 'single')}}, ...
%    'stride', 1, ...
%    'pad', 0) ;

% net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;                      


% net.addLayer(
net.layers{end+1} = struct('type', 'conv', ... %1
   'weights', {{f*randn(3,3,8,5, 'single'), zeros(1, 5, 'single')}}, ...
   'stride', 1,...
   'pad', [1 1 1 1]);
net.layers{end+1} = struct('type', 'relu') ; % 2
net.layers{end+1} = struct('type', 'pool', ... % 3
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [1 1 1 1]) ;
net.layers{end+1} = struct('type', 'conv', ... %4
   'weights', {{f*randn(3,3,5,10, 'single'), zeros(1, 10, 'single')}}, ...
   'stride', 1);
net.layers{end+1} = struct('type', 'pool', ... % 5
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', [1 1 1 1]) ;
net.layers{end+1} = struct('type', 'relu'); % 6
net.layers{end+1} = struct('type', 'conv', ... % 7
   'weights', {{f*randn(3,3,10,5, 'single'), zeros(1, 5, 'single')}}, ...
   'stride', 1, ...
   'pad', [1 1 1 1]);

net.layers{end+1} = struct('type', 'relu') ;      % 8              
net.layers{end+1} = struct('type', 'conv', ... % 9
                           'weights', {{f*randn(3,3,5,8, 'single'), zeros(1, 8, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 1);

net.layers{end+1} = struct('type', 'softmaxloss','name','softmaxlosslayer') ;


% if opts.useBnorm
% net = insertBnorm(net, 1) ;
% net = insertBnorm(net, 5) ;
% net = insertBnorm(net,9) ;
% 

net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;                      


net.removeLayer('softmaxlosslayer');
net.addLayer('objective', ...
  SegmentationLoss('loss', 'softmaxlog'), ...
  {'prediction', 'label'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

% optionally switch to batch normalization
% end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)
% --------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));
ndim = size(net.layers{l}.weights{1}, 4);
layer = struct('type', 'bnorm', ...
               'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
               'learningRate', [1 1], ...
               'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;
