classdef DeepFeatureExtractor_cifar < FeatureExtractor
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
        function obj = DeepFeatureExtractor_cifar(conf)
            obj = obj@FeatureExtractor(conf);
            % load a dnn network...
%             vl_setupnn
            %net = init_nn_network('imagenet-vgg-s.mat');            
            L = load('/home/amirro/code/mircs/data/cifar-lenet/net-epoch-30.mat');
            obj.net = L.net;
            L = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/data/nin_normalization');
            obj.cifar_normalization = L.cifar_normalization;                       
            %net.layers = net.layers(1:32);
%             obj.layers = 16;
            obj.net.layers = obj.net.layers(1:10);                        
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
%             x = normalize_vec(x);
            
        end
        
        function x = extractFeaturesMulti(obj,imgs,dummy)            
            imgs = normalize_data_cifar(imgs,obj.cifar_normalization);
            x = vl_simplenn(obj.net, imgs, [], [], 'disableDropout', true);
            x = squeeze(x(end).x);
        end
    end
end
