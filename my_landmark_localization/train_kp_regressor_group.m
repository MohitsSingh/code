function [ model ] = train_kp_regressor_group(X,kps)
%TRAIN_KP_REGRESSOR Summary of this function goes here
%   Detailed explanation goes here
    
    models = {};
    for z = 1:size(kps,2)        
        models{z} = train_kp_regressor(X,squeeze(kps(:,z,:)));
    end
    models = [models{:}];
    model = struct('w',cat(1,models.w));
%     model.w = [model_x.w;model_y.w];

end

