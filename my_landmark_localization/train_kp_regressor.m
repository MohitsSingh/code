function [ model ] = train_kp_regressor(X,kp)
%TRAIN_KP_REGRESSOR Summary of this function goes here
%   Detailed explanation goes here
    opts_string = '-s 13 -p .1 -B 1';
    model_x = train(kp(:,1), sparse(double(X)), opts_string, 'col');
    model_y = train(kp(:,2), sparse(double(X)), opts_string, 'col');    
    model.w = [model_x.w;model_y.w];

end

