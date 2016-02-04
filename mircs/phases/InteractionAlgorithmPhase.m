classdef InteractionAlgorithmPhase < AlgorithmPhase
    %ALGORITHMPHASE Encapsulation of one phase of the algorithm,
    % including feature extracting, candidate generation and pruning from
    % last phase, etc.
    %   Detailed explanation goes here
    properties
        %regionSampler = RegionSampler();
    end
    methods
        function obj = InteractionAlgorithmPhase(conf,params,featureExtractor)
            obj = obj@AlgorithmPhase(conf,params,featureExtractor);
            obj.params.learningParams.ovpType = 'overlap';
            obj.params.learning.task = 'classification';
            obj.params.learning.include_gt_region = true;
            obj.params.testMode = false;
        end
    end
    
    methods
        function candidates = getCandidates(obj,imgData,prev_candidates)
            candidates = getRegionCandidates(obj.conf,imgData,prev_candidates,obj.params);
        end
        function [regions,labels,ovps] = sampleRegions(obj,regions,groundTruth)
            if isempty(groundTruth)
                labels = -1*ones(size(regions(:)));
                ovps = zeros(size(regions(:)));
            else
                [regions,labels,ovps] = sampleRegions(regions,{groundTruth},obj.params);
            end
        end
        
        function current_candidates = pruneCandidates(obj,imgData,prev_candidates,...
                prev_feats,candidates,prevPhase,imgInd)
            feats = obj.extractFeatures(imgData,candidates)
            curClassifier = pruneCandidatesHelper(prevPhase,imgInd);
            [ws bs] = get_w_from_classifiers(curClassifier);
            candidate_scores = bsxfun(@plus,ws'*feats,bs);
            [regionOvp,ints,uns] = regionsOverlap3(candidates,candidates);
            regionSubset = suppresRegions(regionOvp,.7,candidate_scores,imgData.I_sub);
            current_candidates = candidates(regionSubset);
        end
        
        function feats = extractFeatures(obj,imgData,regions)
            sz = [9 9];
            
            
            
            % extract occupancy map around each facial keypoint            
            
            curLandmarks = bsxfun(@minus,imgData.curLandmarks(:,1:2),imgData.mouthBox(:,1:2));
            %             plotPolygons(curLandmarks,'g+','Linewidth',2);
            %             x2(imgData.I_sub);
                        
            % create a window around each landmark            
            
            lm_boxes = round(inflatebbox(curLandmarks,size(imgData.I_sub,1)/2,'both',true));
%             x2(imgData.I_sub); plotBoxes(lm_boxes);
            
            %             m = multiCrop2(regions,lm_boxes);
            r = {};
            sz_t = [5 5];
            
            R = single(cat(3,regions{:}));
            
            for t = 1:size(lm_boxes)
                m = reshape(imResample(cropper(R,lm_boxes(t,:)),sz_t),[],length(regions));
                r{t} = m;
            end
            feats5 = cat(1,r{:});
            %             for iLandmark = 1:size(curLandmarks,1)
            %
            %             end
            
            
            feats1 = cellfun3(@(x) col(imResample(single(x),sz)),regions,2);
            feats2 = {};
            sz3 = [9 9];
            sz1 = size(imgData.I_sub,1)/2;
            for t = 1:length(regions)
                r = regions{t};
                b = region2Box(r);
                b = round(inflatebbox(b,sz1,'both',true));
                r = cropper(r,b);
                feats2{t} = col(imResample(single(r),sz3));
            end
            feats2 = cat(2,feats2{:});
            %             sz1 = size(imgData.I_sub,1)/2;
            %             feats2 = cellfun3(@(x) col(imResample(single(cropper(x,round(inflatebbox(region2Box(x),sz1,'both',true)))),sz)),...
            %             regions);
            
            nChannels = size(imgData.action_obj,3);
            feats3 = zeros(nChannels,length(regions));
            for iChannel = 1:nChannels
                curChannel = imgData.action_obj(:,:,iChannel);
                for t = 1:length(regions)
                    r = regions{t};
                    feats3(iChannel,t) = mean(curChannel(r));
                end
            end
            feats4 = zeros(nChannels,length(regions));
            %             for iChannel = 1:nChannels
            %                 curChannel = imgData.action_obj(:,:,iChannel);
            %                 for t = 1:length(regions)
            %                     r = regions{t};
            %                     feats4(iChannel,t) = mean(curChannel(r))-mean(curChannel(imdilate(r,ones(5)) & ~r));
            %                 end
            %             end
            
            for iChannel = 1:nChannels
                curChannel = imgData.action_obj(:,:,iChannel);
                for t = 1:length(regions)
                    r = regions{t};
                    feats4(iChannel,t) = mean(curChannel(r))-mean(curChannel(~r));
                end
            end
            
            %             x2(imgData.I_sub);
            %             x2(imgData.action_obj(:,:,2));
            
            
            feats = {feats1,feats2,feats3,feats4,feats5};
            %             feats2 = cellfun3(@(x) col(imResample(single(x),sz)),regions,2);
        end
    end
end

