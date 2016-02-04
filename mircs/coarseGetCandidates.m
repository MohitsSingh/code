% function candidates = coarseGetCandidates(conf,imgData,prev_candidates,params)
% %candidates = coarseGetCandidates(conf,imgData,prev_candidates) obtain
% %candidates for the coarse phase of the action detection task
% regionSampler = RegionSampler();
% % train face-non face area regions.
% I_sub = imgData.I_sub;
% % if params.testMode
% %     regionSampler.boxOverlap = .9;
% % else
% regionSampler.boxOverlap = .5;
% 
% % end
% % define bounding box size
% bb_size = round(size2(I_sub)/3);
% regionSampler.boxSize = bb_size;
% s = bb_size(1);
% r = round(regionSampler.sampleOnImageGrid(I_sub));
% r = r(:,1:4);
% [r ,bads] = clip_to_image(r,I_sub);
% candidates.masks = col(box2Region(r,I_sub));
% end