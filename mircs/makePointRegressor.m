function res = makePointRegressor(feats_l2,sel_train,all_kps,pt_sel)
res = struct;
for t = 1:length(pt_sel)
    
    cur_xy = squeeze(all_kps(:,pt_sel(t),:));
    goods = find(~any(isnan(cur_xy),2));
    g_train = intersect(goods,sel_train);
    %     g_test = intersect(goods,sel_test);
            
    x_model = train(cur_xy(g_train,1),sparse(double(feats_l2(:,g_train))), '-s 12 -B 1 -e .0001','col');
    y_model = train(cur_xy(g_train,2),sparse(double(feats_l2(:,g_train))), '-s 12 -B 1 -e .0001','col');
    res(t).w = [x_model.w;y_model.w];
    res(t).g_train = g_train;
end