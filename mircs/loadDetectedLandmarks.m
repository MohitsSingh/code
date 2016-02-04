function [kp_preds,goods] = loadDetectedLandmarks(conf,imgData,landmarkDir)
if (nargin < 3)
    landmarkDir = conf.landmarks_myDir;
end
curOutPath = j2m(landmarkDir,imgData);
res = load(curOutPath);%,'curKP_global','curKP_local');
if (isfield(res,'res'))
    res = res.res;
end
global_pred = res.kp_global;
local_pred = res.kp_local;
preds = local_pred;
bc1 = boxCenters(global_pred);
bc2 = boxCenters(local_pred);
bc_dist = sum((bc1-bc2).^2,2).^.5;
bad_local = bc_dist > 30;
goods_1= global_pred(:,end) > 2;
local_pred(bad_local,1:4) = global_pred(bad_local,1:4);
goods = goods_1 & ~bad_local;
kp_preds = local_pred;
goods = true(size(kp_preds,1),1);

end