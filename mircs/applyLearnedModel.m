function all_results = applyLearnedModel(conf,fra_db,cur_set,params,featureExtractor,models,coarse_data)
all_results = [];
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
debugging = params.debugging;
for it = 1:length(cur_set)
    t = cur_set(it);
    imgData = fra_db(t);    
    I = getImage(conf,imgData);                        
    params.nSamples = 5;
    gt_graph = get_gt_graph(imgData,params.nodes,params,I);
    gt_graph = gt_graph(1:2);
    
    [roiBoxes,curScores,thetas] = scoreCoarseRois(conf,imgData,coarse_data.params,featureExtractor,coarse_data.w,coarse_data.b);
    [r,ir] = max(curScores);
    startTheta = thetas(ir);
    [configs,scores] = findBestConfiguration(imgData,I,gt_graph,params,featureExtractor,...
        models,startTheta,roiBoxes(ir,:),[]);
   
    [mm,imm] = max(scores);
    if ~debugging
        all_results(t).iImage = t;
        all_results(t).imageID = imgData.imageID;
        all_results(t).class = imgData.class;
        all_results(t).bestConfig = configs(imm);
        all_results(t).bestConfigScore = mm;
    end
    
    if debugging
        %configs = sample_configurations(imgData,I,0,gt_graph,regionSampler,params);
        % % % %     curFeats = configurationToFeats2(I,configs,featureExtractor,params);
        % % % %     curScores = w_config'*cat(2,curFeats{:});
%         curMasks=visualizeConfigurations(I,configs,scores,5,0,boxes_coarse(im,:),...
%             [sprintf('%03.0f_',it),imgData.imageID]);
       curMasks=visualizeConfigurations(I,configs,scores,5,0,[],...
            [sprintf('%03.0f_',it),imgData.imageID]);
    end
end

end

