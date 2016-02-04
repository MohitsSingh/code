function [configs,scores] = findBestConfiguration2(imgData,I,gt_graph,regionSampler,params,featureExtractor,...
    models_parts,models_links,startTheta,cands)
gt_configurations = {};
cur_config = struct('bbox',gt_graph{1}.bbox);%,'mask',gt_graph{1}.roiMask);
cur_config.endPoint = boxCenters(cur_config.bbox);
cur_config.startPoint = cur_config.endPoint;
cur_config.xy = box2Pts(cur_config.bbox);
cur_config.theta = startTheta;
cur_config.score = 0; % no need to score the first one as we're looking for the argmax
faceBox = imgData.faceBox;
scaleFactor = faceBox(4)-faceBox(2);
nSamples = params.nSamples;
restrictAngle = true;
scores = cur_config.score;
configs = {cur_config};
expansionFactor = 5;
for iPart = 2:length(gt_graph)
    [r,ir] = sort(scores,'descend');
    n = length(configs);
    ir = ir(1:min(n,expansionFactor));
    prev_configs = configs(ir);
    configs = {};
    scores = {};
    %
    for iConfig = 1:length(prev_configs) % iterate over candidates of depth -1        
        cur_config = prev_configs{iConfig}; % get configuration to expand
        cur_score = cur_config(end).score;        
        % find overlapping regions...
        % generate new patches around current
        nextRois = sampleAround(cur_config(end),nSamples,scaleFactor,params,I,restrictAngle);
        nextRoiBoxes = cellfun3(@(x) x.bbox,nextRois);
        % find large overlap between candidates and current boxes
        [ovp,int] = boxesOverlap(cands.bboxes,nextRoiBoxes);
        [m,im] = max(ovp,[],2);
        [n,in] = max(int,[],2);
        sel_ = m > .35;
        %displayRegions(I,cands.masks,m);
        my_masks = cands.masks(sel_);
        my_boxes = cands.bboxes(sel_,:);
        prevCenter = round(boxCenters(cur_config(end).bbox));
        prevMask = box2Region([prevCenter prevCenter+1],size2(I));
        Z = bwdist(prevMask);
        min_dists = cellfun3(@(x) min(Z(x)),my_masks);
        sel_ = min_dists <= scaleFactor*.2;
        my_boxes = my_boxes(sel_,:);
        my_masks = my_masks(sel_);
        displayRegions(I,my_masks)                                                
%         nextRois = cellfun2(@(x) {x}, nextRois);
%         curScores = zeros(size(nextRois));
        [partFeats,linkFeats] = getPartFeats(I,cur_config,nextRois,featureExtractor,params);
        curScores = models_parts(iPart).w'*partFeats;
        if (params.interaction_features)
            linkScores = models_links(iPart-1).w'*linkFeats;
            curScores = curScores+linkScores;
        end
                                                            
        %curPartFeatures = configurationToFeats2(I,nextRois,featureExtractor,params);        
       
        %curScores = w_parts{iPart}'*cat(2,curPartFeatures{:});
        % keep scores & configurations
        for iCandidate = 1:length(nextRois)
            R = nextRois{iCandidate};
            R.score = curScores(iCandidate)+cur_score;          
            configs{end+1} = [cur_config,R];
            scores{end+1} = R.score;
        end  
    end
    scores = [scores{:}];

end


% end

% visualizeConfigurations(I,configs,[scores{:}],5,.5);

% if ~isBad
%     gt_configurations{end+1} = cur_config;
% end




% % %     function rois = getCandidateRoisSliding(startPt,scaleFactor,samplingParams,onlyTheta,thetaToKeep)
% % %         
% % %         if nargin < 5
% % %             thetaToKeep = true(size(samplingParams.thetas));
% % %         end
% % %         thetas = samplingParams.thetas(thetaToKeep);
% % %         lengths = samplingParams.lengths*scaleFactor;
% % %         widths = samplingParams.widths*scaleFactor;
% % %         if onlyTheta
% % %             lengths = max(lengths);
% % %             widths = mean(widths);
% % %         end
% % %         all_rois = {};
% % %         
% % %         for iLength = 1:length(lengths)
% % %             for iWidth = 1:length(widths)
% % %                 curLength = lengths(iLength);
% % %                 curWidth = widths(iWidth);
% % %                 if curWidth > curLength
% % %                     continue
% % %                 end
% % %                 all_rois{end+1} = hingedSample(startPt,curWidth,curLength,thetas);
% % %             end
% % %         end
% % %         
% % %         rois = cat(2,all_rois{:});
% % %         %     roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
% % %         % roiPatches = cellfun2(@(x) rectifyWindow(I,round(x.xy),[avgLength avgWidth]),rois);
