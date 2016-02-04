
classdef DeepFeatureExtractor < FeatureExtractor
    %FeatureExtractor Extracts features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        net
        layers
        useGPU
    end
    methods
        function obj = DeepFeatureExtractor(conf,useGPU,layer,netPath)
            
            obj = obj@FeatureExtractor(conf);
            if nargin == 1
                obj.useGPU = false;
            else
                obj.useGPU = useGPU;
            end
            % load a dnn network...
            vl_setupnn
            %             if layer==33 % sort of hack
            if nargin < 4
                netPath= '/home/amirro/storage/matconv_data/imagenet-vgg-verydeep-16.mat';
            end
            net_deep = init_nn_network(netPath);
            
            %             else
            %                 %net_deep = init_nn_network('/home/amirro/storage/matconv_data/imagenet-vgg-s.mat');
            %                 net_deep = init_nn_network('/home/amirro/storage/matconv_data/imagenet-vgg-f.mat');
            %             end
            net_deep.normalization.averageImage = single(net_deep.normalization.averageImage);
            for t = 1:length(net_deep.layers)
                curLayer = net_deep.layers{t};
                if (isfield(curLayer,'weights'))
                    curLayer.filters = curLayer.weights{1};
                    curLayer.biases = curLayer.weights{2};
                    curLayer = rmfield(curLayer,'weights');
                end
                
                if isfield(curLayer,'filters') && obj.useGPU
                    curLayer.filters = gpuArray(curLayer.filters)
                    curLayer.biases = gpuArray(curLayer.biases)
                end
                
                net_deep.layers{t} = curLayer;
            end
            if nargin < 3
                %layer = 17;
                layer = 17;
                
                
                
            end
            net_deep.layers = net_deep.layers(1:layer-1);
            obj.layers = layer;
            %net_deep.layers = net_deep.layers(1:layer(end));
            obj.net = net_deep;
        end
        
        function x = extractFeaturesHelper(obj,currentID,regions)
            I = getImage(obj.conf,currentID);
            % get bounding boxes for all regions, unless they're already
            % boxes
            if (iscell(regions))
                boxes = cellfun2(@region2Box,regions);
                boxes = cat(1,boxes{:});
            end
            subImages = multiCrop2(I,round(boxes));
            x = extractDNNFeats(subImages,obj.net,obj.layers,false,false,obj.useGPU);
            x = gather(x.x);
        end
        
        function x = extractFeaturesMulti_mask(obj,img,masks,toMute)
            x = [];
            if isempty(masks)
                return
            end
            if nargin < 4
                toMute = false;
            end
            if ~iscell(masks)
                masks = box2Region(masks,size2(img));
                if ~iscell(masks), masks = {masks}; end
            end
            
            I = {};
            fillValue = .5;
            if isa(img,'uint8')
                fillValue = 128;
            end
            
            for t = 1:length(masks)
                %r = img;
                %curMask = masks{t};
                I{t} = maskedPatch(img,masks{t},true,fillValue);
                %                 m = repmat(~curMask,[1 1 3]);
                %                 r(m) = fillValue;
                %                 curMaskBox = round(makeSquare(region2Box(curMask),true));
                %                 r = cropper(r,curMaskBox);
                %                 I{t} = r;
            end
            x = gather(obj.extractFeaturesMulti(I,toMute));
        end
        
        function x = extractFeaturesMulti(obj,imgs,toMute,rotation)
            if nargin < 3
                toMute = false;
            end
            if nargin < 4
                rotation = 0;
            end
            
            % find out if this is a variable sized cell array
            
            if ~iscell(imgs) || ~iscell(imgs{1}) || (iscell(imgs) &&  max(cellfun(@length,imgs)) == 1)
                x = extractDNNFeats(imgs,obj.net,obj.layers,false,toMute,obj.useGPU,rotation);
                %                 x
                
                x = gather(squeeze(cat(1,x.x)));
            else
                x = {};
                for t = 1:length(imgs)
                    t
                    M = imgs{t};
                    xx = extractDNNFeats(M,obj.net,obj.layers,false,toMute,obj.useGPU,rotation);
                    x{t} = gather(squeeze(cat(1,xx.x)));
                    
                end
            end
        end
        function x = my_extract_dnn_feats_simple(obj,imgs,tofliprgb,keepAspect)
            x = {};
            net = obj.net;
            batchSize = 16;
            if nargin < 3
                tofliprgb = false;
            end
            if nargin < 4
                keepAspect = true;
            end
            batches = batchify(length(imgs),ceil(length(imgs)/batchSize));
            tic_id = ticStatus('extracting deep features',.2,.1);
            
            for iBatch = 1:length(batches)
                curBatch = batches{iBatch};
                v = {};
                for iCurBatch = 1:length(curBatch)
                    k = curBatch(iCurBatch);
                    im = imgs{k};
                    if ischar(im)
                        im = imread(im);
                        %                         im = uint8(im{1});
                    end
                    if size(im,3)==1
                        im = cat(3,im,im,im);
                    end
                    if tofliprgb
                        %                         im = cat(3,im(:,:,1)',im(:,:,2)',im(:,:,3)');
                        im = im(:,:,[3 2 1]);
                    end
                    im_ = single(im); % note: 255 range
                    
                    if ~keepAspect
                        im_ = imResample(im_, net.normalization.imageSize(1:2),'bilinear');
                    else
                        %                         resizeRatio = net.normalization.imageSize(1:2)./size2(im_);
                        squareImageSize = max(size2(im_));
                        z = true(size(im_));
                        diffs = floor((squareImageSize-size2(im_))/2);
                        im_ = padarray(im_,diffs,0,'both');
                        z = padarray(z,diffs,0,'both');
                        im_ = imResample(im_, net.normalization.imageSize(1:2),'bilinear');
                        z = imResample(z,size2(im_),'nearest');
                        im_(~z) = net.normalization.averageImage(~z);
                        
                    end
                    im_ = im_ - net.normalization.averageImage ;
                    v{iCurBatch} = im_;
                end
                imo = cat(4,v{:});
                imo = gpuArray(imo);
                opts.disableDropout = false;
                dnn_res = vl_simplenn(net, imo);
                r = gather(dnn_res(obj.layers(end)).x);
                x{end+1} = reshape(squeeze(r),[],length(curBatch));
                tocStatus(tic_id,iBatch/length(batches));
                %         reshape(dnn_res(layers(iLayer)).x,[],batchSize);
            end
            x = cat(2,x{:});
        end
        function x = my_extract_dnn_feats_multiscale(obj,imgs,scales,toMute)
            x = {};
            net = obj.net;
            batchSize = 16;
            
            if nargin < 3
                scales = [256 384 512];
            end
            if nargin < 4
                toMute=true;
            end
            if ~toMute
                tic_id = ticStatus('extracting deep features',1,.1);
            end
            flips = [false true];
            for iImage = 1:length(imgs)
                
                I = imgs{iImage};
                if isempty(scales)
                    scales = min(size2(I,1));
                end
                if ischar(I)
                    I = imread(I);
                end
                if size(I,3)==1
                    I = cat(3,I,I,I);
                end
                I = single(I);
                mean_value = net.normalization.averageImage(1,1,:);
                m = min(size2(I));
                
                
                n = 0;
                curSum = zeros(1,1,4096);
                scaleFactors = scales/m;
                maxSizesAfterScaling =  max(size2(I)).*scaleFactors;
                
                scales_ = scales((prod(size2(I))^.5)*scaleFactors < 1100);
                
                for iFlip = 1:length(flips)
                    if flips(iFlip)
                        I = flip_image(I);
                    end
                    for iSize = 1:length(scales_)                                                                                              
                        II = imResample(I,scales_(iSize)/m);
                        ss = size2(I);
                        if (min(ss) < size2(net.normalization.averageImage,1))
                            s_pad = max(size2(net.normalization.averageImage,1)-ss,0);
                            diffs_pre = floor(s_pad/2);
                            diffs_post = s_pad-diffs_pre;
                            z = true(size2(I));
                            II = padarray(II,diffs_pre,0,'pre');
                            II = padarray(II,diffs_post,0,'post');
                            z = padarray(z,diffs_pre,0,'pre');
                            z = padarray(z,diffs_post,0,'post');
                            for iz = 1:size(II,3)
                                c_ = II(:,:,iz);
                                c_(~z) = net.normalization.averageImage(1,1,iz);
                                II(:,:,iz) = c_;
                            end
                        end
                        II = bsxfun(@minus,II,mean_value);
                        %                         size2(II)
                        if obj.useGPU
                            II = gpuArray(II);
                            opts = struct('conserveMemory',true);
                            dnn_res = vl_simplenn(net, II,[],opts);
                            clear II;
                        else
                            dnn_res = vl_simplenn(net, (II));
                        end
                        r = double(gather(dnn_res(obj.layers(end)).x));
                        curSum = curSum+sum(sum(r,1),2);
                        n = n+prod(size2(r));
                    end
                end
                
                curSum = curSum(:)/n;
                x{iImage} = curSum;
                if ~toMute
                    tocStatus(tic_id,iImage/length(imgs));
                end
            end
            x = cat(2,x{:});
        end
        %             end
    end
end

