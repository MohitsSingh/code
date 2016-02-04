function candidates = coarsePruneCandidates( imgData,prev_candidates,prev_feats,candidates,phase,imgInd)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    I_sub = imgData.I_sub;
    
    curClassifier = pruneCandidatesHelper(phase,imgInd);
    [ws bs] = get_w_from_classifiers(curClassifier);
%     feats = phase.featureExtractor.extractFeatures(I_sub,prev_candidates);
    scores = bsxfun(@plus,ws'*prev_feats,bs);   
    heatMap = computeHeatMap_regions(I_sub,prev_candidates,scores,'max');                
    candidate_scores = cellfun3(@(x) sum(heatMap(x(:))),candidates);
    areas = cellfun3(@nnz,candidates);    
    candidate_scores = candidate_scores./areas;
%     displayRegions(I_sub,candidates,candidate_scores);
    
    [regionOvp,ints,uns] = regionsOverlap3(candidates,candidates);
    regionSubset = suppresRegions(regionOvp,.7,candidate_scores,I_sub);
    candidates = candidates(regionSubset);
%     displayRegions(I_sub,candidates(regionSubset),candidate_scores(regionSubset));
%     x2(heatMap);
% z = 1;
end

