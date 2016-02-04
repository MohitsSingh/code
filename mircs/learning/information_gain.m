function [gain_curve,p_left,p_right] = information_gain(p)
%INFORMATION_GAIN calculate the threshold to split the vector p 
% to attain the maximal information gain, where p is an indicator function
% for two classes.

p = double(p(:)'); % make it a row vector

size_left = (1:length(p));
size_right = (length(p):-1:1);

% num of p==1 in left and right sides

p_left = cumsum(p)./size_left;
p_right = fliplr(cumsum(fliplr(p)))./size_right;

log_left = log(p_left);
log_right = log(p_right);
log_left(p_left==0) = 0;
log_right(p_right==0) = 0;
ent_left = -p_left.*log_left - (1-p_left).*log(1-p_left);
ent_right = -p_right.*log_right - (1-p_right).*log(1-p_right);
ent_left(isnan(ent_left)) = 0;
ent_right(isnan(ent_right)) = 0;


gain_curve = (-size_left.*ent_left-size_right.*ent_right)/length(p);
gain_curve(isnan(gain_curve)) = min(gain_curve(~isnan(gain_curve)));
end

