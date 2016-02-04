function net = init_nn_network(networkPath)
vl_setupnn
if (nargin == 0)
    networkPath = '~/storage/ imagenet-vgg-s.mat';
end

net = load(networkPath);
for t = 1:length(net.layers)
    curLayer = net.layers{t};
    if (isfield(curLayer,'weights'))
        curLayer.filters = curLayer.weights{1};
        curLayer.biases = curLayer.weights{2};
        curLayer = rmfield(curLayer,'weights');
        net.layers{t} = curLayer;
    end
end