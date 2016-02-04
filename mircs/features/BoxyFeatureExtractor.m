
classdef BoxyFeatureExtractor < FeatureExtractor
    %FeatureExtractor Extracts features from regions or windows.
    %   Detailed explanation goes here
    
    properties
        internalFeatureExtractor
        %net
        %layers
        %useGPU
        
    end
    methods
        function obj = BoxyFeatureExtractor(conf,internalFeatureExtractor)            
            obj = obj@FeatureExtractor(conf);            
        end        
        function x = extractFeaturesHelper(obj,currentID,regions)                        
            I = getImage(obj.conf,currentID);
            % get bounding boxes for all regions, unless they're already
            % boxes
            if (iscell(regions))
                boxes = cellfun2(@region2Box,regions);
                boxes = cat(1,boxes{:});
            end
            
            % find for each box the box which overlaps it the most from the
            % discrete set of boxes in this image. Assign to it the
            % features from that box 
            
            
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
        
        function x = extractFeaturesMulti(obj,imgs,toMute)
            if nargin < 3
                toMute = false;
            end
            x = extractDNNFeats(imgs,obj.net,obj.layers,false,toMute,obj.useGPU);
            x = gather(squeeze(x(end).x));
        end
    end
end
