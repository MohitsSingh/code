classdef FineAlgorithmPhase < AlgorithmPhase
    %ALGORITHMPHASE Encapsulation of one phase of the algorithm,
    % including feature extracting, candidate generation and pruning from
    % last phase, etc.
    %   Detailed explanation goes here
    properties
        %regionSampler = RegionSampler();
    end    
    methods
        function obj = FineAlgorithmPhase(conf,params,featureExtractor)
            obj = obj@AlgorithmPhase(conf,params,featureExtractor);
            obj.params.learningParams.ovpType = 'overlap';
            obj.params.learning.task = 'classification';
            obj.params.learning.include_gt_region = true;
            obj.params.testMode = false;
        end
    end
    
    methods
    end
    methods
        
        %% % phases(nPhase).getCandidates = @getRegionCandidates;
        % phases(nPhase).sampleRegions= @sampleRegions;
        % phases(nPhase).extractFeatures = @fineExtractFeatures;
        %%
        
        function candidates = getCandidates(obj,imgData,prev_candidates)
            candidates = getRegionCandidates(obj.conf,imgData,prev_candidates,obj.params);
        end
        function [regions,labels,ovps] = sampleRegions(obj,regions,groundTruth)
            [regions,labels,ovps] = sampleRegions(regions,{groundTruth},obj.params);
        end
        function feats = extractFeatures(obj,imgData,regions)
            feats_app = obj.featureExtractor.extractFeaturesMulti_mask(imgData.I_sub,regions,true);                        
            mask_images = cellfun2(@(x) single(cat(3,x,x,x)),regions);
            feats_shape = obj.featureExtractor.extractFeaturesMulti(mask_images,true);            
            feats = {feats_app,feats_shape};            
        end
    end
end

