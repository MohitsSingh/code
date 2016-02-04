function [candidates,ucm2,isvalid] = getCandidateRegions(conf,imgData,I_sub,isTrain)

isvalid = true;

load(j2m('~/storage/fra_db_mouth_seg_2',imgData)); % candidates, ucm2

if isTrain
    r = segs([segs.useGT]==1);
else
    r = segs([segs.useGT]==0);
end
candidates = r.candidates;
ucm2 = r.ucm2;

% if ~r.success
%     isvalid = false;
    
masks_sub = candidates.masks;
nonZeros = squeeze(sum(sum(masks_sub,1),2));
goods = nonZeros > 0 & nonZeros < prod(size2(masks_sub(:,:,1)));
z= {};
for u = 1:length(masks_sub)
    if goods(u)
        z{end+1} = masks_sub(:,:,u);
    end
end
candidates.masks= row(z);
candidates.bboxes = candidates.bboxes(goods,[2 1 4 3]);
