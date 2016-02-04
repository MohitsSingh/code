function [sel_train,sel_val,all_kps] = makeTrainingSet(imgs_in,kp_in)
sel_train = 1:2:length(imgs_in);
sel_val = 2:2:length(imgs_in);
all_kps = zeros(size(kp_in,1),1,2);
all_kps(:) = kp_in;
end
