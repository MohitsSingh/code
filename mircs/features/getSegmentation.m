function [candidates,ucm,segBox,isvalid] = getSegmentation(conf,imgData,convertToMasks,segPath)
%R = j2m('~/storage/fra_face_seg',imgData);
if (nargin < 4)
    segPath = conf.face_seg_dir;
end    
R = j2m(segPath,imgData);
L1 = load(R);
if (~isfield(L1,'res'))
    tmp = struct('res',L1);
    L1 = tmp;
end
ucm = L1.res.ucm2;
segBox = [];
isvalid = ~isfield(L1.res,'valid') || L1.res.valid;

if (~isvalid)
    return;
end
if (nargin < 3)
    convertToMasks = false;
end
candidates = L1.res.candidates;
if (convertToMasks)
    candidates.masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
    candidates.masks = squeeze(mat2cell2(candidates.masks,[1 1 size(candidates.masks,3)]));
end
segBox = L1.res.roiBox;
end