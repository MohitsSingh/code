function candidates = getRegionCandidates(conf,imgData,prev_candidates,params)
%candidates = coarseGetCandidates(conf,imgData,prev_candidates) obtain
%candidates for the coarse phase of the action detection task
% ignore the previous for now...
I_sub = imgData.I_sub;
mouthMask = imgData.mouthMask;
[candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub,~params.testMode);
candidates.masks = processRegions(I_sub,candidates,mouthMask); % remove some obviously bad regions;
candidates.masks = col(candidates.masks);
end