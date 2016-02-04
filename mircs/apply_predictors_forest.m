function preds_xy = apply_predictors_forest(predictors,feats_l2,cur_set)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% wx = cat(1,predictors.wx);
% wy = cat(1,predictors.wy);

preds_xy = zeros(size(feats_l2(:,cur_set),2),length(predictors),2);
for t = 1:length(predictors)
    px = fernsRegApply(double(feats_l2(:,cur_set)'),predictors(t).fernsx);
    py = fernsRegApply(double(feats_l2(:,cur_set)'),predictors(t).fernsy);
    preds_xy(:,t,:) = [px py];
%     preds{t} = 
end

% predicted_kp_x = predict2(feats_l2(:,cur_set),wx);
% predicted_kp_y = predict2(feats_l2(:,cur_set),wy);

% %%
% preds_xy = zeros(length(cur_set),length(predictors),2);
% for ip = 1:length(predictors)
%     preds_xy(:,ip,:) = [predicted_kp_x(:,ip) predicted_kp_y(:,ip)];
% end
% 
end

