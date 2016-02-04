function predictors = train_predictors_forest(feats_l2,sel_train,all_kps,kpNames,lambda)
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
    %     kp_x_model = train(kp_xy(g_train,1),sparse(double(feats_l2(:,g_train))), trainOpts,'col');
    %     kp_y_model = train(kp_xy(g_train,2),sparse(double(feats_l2(:,g_train))), trainOpts,'col');
    prm=struct('type','res','loss','L1','eta',.05,...
   'thrr',[-1 1],'reg',.01,'S',2,'M',2048,'R',3,'verbose',0);    

    xs0 = double(feats_l2(:,g_train))';
    ys0 = kp_xy(g_train,1);
    [fernsx] = fernsRegTrain(xs0,ys0,prm);
   
    ys0 = kp_xy(g_train,2);
    [fernsy] = fernsRegTrain(xs0,ys0,prm);
    
    predictors(t).fernsx = fernsx;
    predictors(t).fernsy = fernsy;
end