function [preds,goods] = getKPPredictions(L,t)

local_pred = squeeze(L.all_kp_predictions_local(t,:,:));
global_pred = squeeze(L.all_kp_predictions_global(t,:,:));
goods_1= global_pred(:,end) > 2;

%     R = j2m('~/storage/fra_face_seg',fra_db(t));
%     L1 = load(R);
%     %     preds = (local_pred+global_pred)/2;
%     candidates = L1.res.candidates;
%     candidates.masks = cands2masks(candidates.cand_labels, candidates.f_lp, candidates.f_ms);
%     candidates.masks = squeeze(mat2cell2(candidates.masks,[1 1 size(candidates.masks,3)]));
%     displayRegions(I,candidates.masks,[],0);
preds = local_pred;
bc1 = boxCenters(global_pred);
bc2 = boxCenters(local_pred);
bc_dist = sum((bc1-bc2).^2,2).^.5;
bad_local = bc_dist > 30;
preds(bad_local,1:4) = global_pred(bad_local,1:4);
goods = goods_1 & ~bad_local;
