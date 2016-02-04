classdef CoarseAlgorithmPhase < AlgorithmPhase
    %ALGORITHMPHASE Encapsulation of one phase of the algorithm,
    % including feature extracting, candidate generation and pruning from
    % last phase, etc.
    %   Detailed explanation goes here
    
    properties
        regionSampler
        gt_regions        
    end
    
    methods
        function obj = CoarseAlgorithmPhase(conf,params,featureExtractor,gt_regions)
            obj = obj@AlgorithmPhase(conf,params,featureExtractor);
            obj.params.learning.ovpType = 'overlap';
            obj.params.learning.posOvp = .5;
            obj.params.learning.negOvp = .2;
            obj.params.learning.include_gt_region = true;
            obj.params.learning.task = 'regression';
            obj.regionSampler = RegionSampler();
        end
    end
    methods
        
        function candidates = getCandidates(obj,imgData,prev_candidates)
            % train face-non face area regions.
            I_sub = imgData.I_sub;
            %action_region = imgData.action_obj == 2;
            
            [candidates,ucm2,isvalid] = getCandidateRegions(obj.conf,imgData,I_sub,~obj.params.testMode);
            candidates = candidates.masks;            
            candidates = col(ezRemove(candidates,I_sub,50,.3));
        end
        
        
        function candidates = getCandidates_old(obj,imgData,prev_candidates)
            % train face-non face area regions.
            I_sub = imgData.I_sub;
            action_region = imgData.action_obj.active == 2;
            
            if none(action_region)
                candidates = [];
                return
            end
            
            [L,numComponents] = bwlabel(action_region);
            a = {};
            for t = 1:numComponents
                if nnz(L==t)>10
                    a{end+1} = L==t;
                end
            end
            action_region = a;
            
%             if obj.params.testMode            
            %mouthMask = imgData.mouthMask;
            [candidates,ucm2,isvalid] = getCandidateRegions(obj.conf,imgData,I_sub,~obj.params.testMode);
            candidates = candidates.masks;            
            candidates = row(ezRemove(candidates,I_sub,50,.3));
            %candidates{end+1} = action_region;
            candidates = [candidates,action_region(:)'];                
            [ovp,ints,uns] = regionsOverlap(candidates,action_region,false);
            ovp = max(ovp,[],2);
            candidates = col(candidates(ovp>.15));
            ovp = ovp(ovp>.15);
        end
        function candidates = pruneCandidates(obj,imgData,prev_candidates,...
                prev_feats,candidates,prevPhase,imgInd)
            I_sub = imgData.I_sub;
            curClassifier = pruneCandidatesHelper(prevPhase,imgInd);
            [ws bs] = get_w_from_classifiers(curClassifier);
            scores = bsxfun(@plus,ws'*prev_feats,bs);
            heatMap = computeHeatMap_regions(I_sub,prev_candidates,scores,'max');
            candidate_scores = cellfun3(@(x) sum(heatMap(x(:))),candidates);
            areas = cellfun3(@nnz,candidates);
            candidate_scores = candidate_scores./areas;
            %     displayRegions(I_sub,candidates,candidate_scores);
            [regionOvp,ints,uns] = regionsOverlap3(candidates,candidates);
            regionSubset = suppresRegions(regionOvp,.7,candidate_scores,I_sub);
            candidates = candidates(regionSubset);
        end
        function [regions,labels,ovps] = sampleRegions(obj,regions,groundTruth)
            [regions,labels,ovps] = sampleRegions(regions,{groundTruth},obj.params);
        end
        function feats = extractFeatures(obj,imgData,regions)
            %feats = obj.featureExtractor.extractFeaturesMulti_mask(imgData.I_sub,regions);
        end
    end
end

