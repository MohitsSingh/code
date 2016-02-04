function [gain_curve] = information_gain3(p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
p = double(p(:)'); % make it a row vector
gain_curve = zeros(size(p));
s_left = 1:length(gain_curve);
s_right =length(gain_curve)-1:-1:1;
p_left = cumsum(p)./s_left;
p_right = fliplr(cumsum(fliplr([p(2:end)])))./s_right;
log_left = log(p_left);
log_right = log(p_right);
log_left(p_left==0) = 0;
log_right(p_right==0) = 0;
ent_left = -p_left.*log_left-(1-p_left).*log(1-p_left);
ent_right = -p_right.*log_right-(1-p_right).*log(1-p_right);
ent_left(isnan(ent_left)) = 0;
ent_right(isnan(ent_right)) = 0;
gain_curve(1:end-1) = [(-s_left(1:end-1).*ent_left(1:end-1)-[ s_right].*[ ent_right])/length(p)];
gain_curve(end) = min(gain_curve);

end

