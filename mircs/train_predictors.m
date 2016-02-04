function predictors = train_predictors(feats_l2,sel_train,all_kps,kpNames,lambda)
if nargin < 5
    lambda = .1;
end
predictors = struct('kp_name',{},'kp_ind',{},'wx',{},'wy',{});
% toCheck = [1 2];
trainOpts = sprintf('-s 11 -B 1 -e %f',lambda);
for t = 1:length(kpNames)
    t
    predictors(t).kp_name = kpNames{t};
    kp_xy = squeeze(all_kps(:,t,:));
    goods = find(~any(isnan(kp_xy),2));
    g_train = intersect(goods,sel_train);
    %kp_x_model = train(kp_xy(g_train,1),sparse(double(feats_l2(:,g_train))), '-s 11 -B 1 -e .00001','col');
    %kp_y_model = train(kp_xy(g_train,2),sparse(double(feats_l2(:,g_train))), '-s 11 -B 1 -e .00001','col');
    kp_x_model = train(kp_xy(g_train,1),sparse(double(feats_l2(:,g_train))), trainOpts,'col');
    kp_y_model = train(kp_xy(g_train,2),sparse(double(feats_l2(:,g_train))), trainOpts,'col');
    predictors(t).wx = single(kp_x_model.w);
    predictors(t).wy = single(kp_y_model.w);
end