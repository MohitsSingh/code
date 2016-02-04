function features = collectTrainingSamples(conf,fra_db,trainingSets,params,featureExtractor,prevDetector)%,coarse_data)
cur_set = trainingSets.sel_pos;
nodes = params.nodes;
if ~isfield(params,'debugging');
    debugging = false;
else
    debugging = params.debugging;
end

if nargin < 6
    prevDetector = [];
end
if nargin < 7 
    coarse_data = [];
end

for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    gt_graph = gt_graph(1:2);
    if (any(cellfun(@(x) ~isfield(x,'roiMask') || nnz(x.roiMask)==0 ,gt_graph)))
        warning('skipping due to missing element...')
        continue
    end
    %     continue
    %displayRegions(I,cellfun2(@(x) x.roiMask,gt_graph))
    min_gt_ovp = .5;
    if any(sum(cellfun3(@(x) x.bbox,gt_graph,1),2)==0)
        disp('skipping current example since some nodes are missing');
        continue
    end
    myParams = params;
    myParams.restrictAngle = false;
    gt_configurations = sample_configurations(imgData,I,min_gt_ovp,gt_graph,params);
    if debugging
        visualizeConfigurations(I,gt_configurations);
        continue
    end
    [curPartFeats,curIntFeats,curShapeFeats] = configurationToFeats2(I,gt_configurations,featureExtractor,params);
    pos_feats(it).partFeats = curPartFeats;
    pos_feats(it).intFeats = curIntFeats;
    pos_feats(it).shapeFeats = curShapeFeats;
    %all_pos_feats{end+1} =
end
%
% negative samples
cur_set = trainingSets.sel_neg;
debugging = false;
neg_feats = struct;
params.conf = conf;
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    params.nSamples = 5;
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    gt_graph = gt_graph(1:2);
      
    
    if ~isempty(prev_detector)
        
        all_results = applyLearnedModel(conf,fra_db,cur_set,params,featureExtractor,models,coarse_data)
    end
    % if we already have a prior detector, use it to sample configurations                
% %     if ~isempty(coarse_data)                        
% %         [rois,curScores,thetas] = scoreCoarseRois(conf,imgData,coarse_data.params,featureExtractor,coarse_data.w,coarse_data.b);
% %         [r,ir] = max(curScores);
% %     end
%     configs = sample_configurations(imgData,I,0,gt_graph,params,thetas(ir));
    if debugging
        visualizeConfigurations(I,configs);
        %         dpc
    end
    [curPartFeats,curIntFeats,curShapeFeats] = configurationToFeats2(I,configs,featureExtractor,params);
    neg_feats(it).partFeats = curPartFeats;
    neg_feats(it).intFeats = curIntFeats;
    neg_feats(it).shapeFeats = curShapeFeats;
end

features.pos_feats = pos_feats;
features.neg_feats = neg_feats;