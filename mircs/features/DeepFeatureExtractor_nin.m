classdef DeepFeatureExtractor_nin < FeatureExtractor
    %FeatureExtractor Extracts features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        net
        layers
    end
    properties (Access = private)
        cifar_normalization
        
    end
    
    methods
        function obj = DeepFeatureExtractor_nin(conf)
            obj = obj@FeatureExtractor(conf);
            % load a dnn network...
            ninet = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/data/cifar-nin/net-epoch-44.mat');
            load /home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/data/nin_normalization
            obj.cifar_normalization = cifar_normalization;
            net = ninet.net;
            net.layers = net.layers(1:end-1);
            
            %net_deep = init_nn_network('imagenet-vgg-s.mat');
            
            %             for t = 1:length(net_deep.layers)
            %                 curLayer = net_deep.layers{t};
            %                 if (isfield(curLayer,'weights'))
            %                     curLayer.filters = curLayer.weights{1};
            %                     curLayer.biases = curLayer.weights{2};
            %                     curLayer = rmfield(curLayer,'weights');
            %                     net_deep.layers{t} = curLayer;
            %                 end
            %             end
            %net_deep.layers = net_deep.layers(1:32);
            obj.layers = 20;
            net.layers = net.layers(1:obj.layers-1);
            obj.net = net;
        end
        
        function x = extractFeaturesHelper(obj,currentID,regions)
            I = getImage(obj.conf,currentID);
            % get bounding boxes for all regions, unless they're already
            % boxes
            if (iscell(regions))
                 if (length(regions)==1 && size(regions{1},2)==4)
                     boxes = regions{1};
                 else
%                 if (numel(regions{1})==4)                    
                    boxes = cellfun2(@region2Box,regions);
                    boxes = cat(1,boxes{:});                
                 end
            else
                boxes = regions;
            end
            subImages = multiCrop2(I,round(boxes));
            subImages = normalize_data_cifar(subImages,obj.cifar_normalization);
            x = vl_simplenn(obj.net, subImages, [], [], 'disableDropout', true);
            x = squeeze(x(end).x);
        end
        function x = extractFeaturesMulti(obj,imgs,dummy)
            imgs = normalize_data_cifar(imgs,obj.cifar_normalization);
            x = vl_simplenn(obj.net, imgs, [], [], 'disableDropout', true);
            x = squeeze(x(end).x);
        end
    end
end
