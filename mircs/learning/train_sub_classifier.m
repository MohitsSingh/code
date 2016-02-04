
function [w,b] = train_sub_classifier(feat_train,t_train,sel)
% convenience function for training a classifier for a subclass

it_train = find(t_train); % original true labels
if (nargin < 3 || isempty(sel))
    sel = 1:length(it_train);
end
t_train = -ones(size(t_train)); % new true labels - set old ones to neutral/don't care
sel_c = setdiff(it_train,it_train(sel));
t_train(it_train(sel)) = 1; % 
t_train(sel_c) = 0;

[w,b] = train_classifier(feat_train(:,t_train==1),feat_train(:,t_train==-1),.1,10);

