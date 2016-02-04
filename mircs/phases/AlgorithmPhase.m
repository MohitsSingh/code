classdef AlgorithmPhase < handle
    %ALGORITHMPHASE Encapsulation of one phase of the algorithm,
    % including feature extracting, candidate generation and pruning from
    % last phase, etc.
    %   Detailed explanation goes here
    
    properties
        conf
        params
        featureExtractor
        classifiers
    end
    
    methods
        function obj = AlgorithmPhase(conf,params,featureExtractor)
            obj.conf = conf;
            obj.params = params;
            obj.classifiers = [];
            if nargin == 3
                obj.featureExtractor = featureExtractor;
            end
        end
        function obj = setTestMode(obj,m)
             obj.params.testMode = m;
        end
    end
    
    methods
        function current_candidates = pruneCandidates(obj,imgData,prev_candidates,...
                prev_feats,candidates,prevPhase,imgInd)
            % do nothing by default
        end
    end
    methods(Abstract)
        getCandidates(obj,imgData,prev_candidates)
        
        sampleRegions(obj,candidates,groundTruth)
        extractFeatures(obj,candidates)
    end
end

