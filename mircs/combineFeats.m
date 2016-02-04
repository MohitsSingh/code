function [f] = combineFeats(feats,varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%     feats 1 - global shape
% feats 2 - local shape
% feats 3 - probs
% feats 4 - prob difference
% feat 5 - local interaction
% feats 6 - apperance dnn
% feats 7 - shape dnn
    
%     f_ = f1;%     
    
    global_shape = cat(2,feats{:,1});
%     f1 = normalize_vec(f1);
    global_shape = vl_homkermap(global_shape,1);
    
    local_shape = cat(2,feats{:,2});
%     f2 = normalize_vec(f2);
    local_shape = vl_homkermap(local_shape,1);

    probs = cat(2,feats{:,3});
%     f3 = normalize_vec(f3);
    probs = vl_homkermap(probs,1);
    
    probs_diff = cat(2,feats{:,4});
%     f4 = normalize_vec(f4);
    probs_diff = vl_homkermap(probs_diff,1);
%     f0 = [f0;f1;f2;f3];
    local_interaction = cat(2,feats{:,5});
    local_interaction = normalize_vec(local_interaction);
%     f4 = normalize_vec(f4);
%     f5 = vl_homkermap(f5,1);
    
    %f = [f1;f2;f3;f4;f5];
    
    app_dnn = cat(2,feats{:,6});
    shape_dnn = cat(2,feats{:,7});

    f = [global_shape;local_shape;probs;probs_diff;local_interaction;app_dnn;shape_dnn];
%     f = f5;
%     f0 = [
%     f = [f3];%f3;
%     f1 = normalize_v
%     f = vl_homkermap(normalize_vec([f0;f1]),1);
    
%     f = [f0;f1];
    
%     f2 = cat(2,feats{:,2});
%     f = [f1;f2/10];
%     f = [f0;f1];

end

