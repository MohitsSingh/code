function [gain_curve] = information_gain2(p)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
p = double(p(:)'); % make it a row vector
gain_curve = zeros(size(p));



for k = 1:length(gain_curve)
    s_left = k;
    s_right = length(gain_curve)-k;
    p_left = sum(p(1:k))/s_left;
    p_right = sum(p(k+1:end))/s_right;
    log_left = log(p_left);
    log_right = log(p_right);
    log_left(p_left==0) = 0;
    log_right(p_right==0) = 0;
    
    ent_left = -p_left*log_left-(1-p_left)*log(1-p_left);
    ent_right = -p_right*log_right-(1-p_right)*log(1-p_right);
    
    ent_left(isnan(ent_left)) = 0;
    ent_right(isnan(ent_right)) = 0;
    % TODO- information gain should be the negative of this. But the best
    % results actually show the reverse.... need to check out why this
    % happens. 
    gain_curve(k) = [(-s_left.*ent_left-s_right.*ent_right)/length(p)];
end





% size_left = (1:length(p));
% size_right = (length(p):-1:1);
% 
% % num of p==1 in left and right sides
% 
% p_left = cumsum(p)./size_left;
% p_right = fliplr(cumsum(fliplr(p)))./size_right;
% 
% log_left = log(p_left);
% log_right = log(p_right);
% log_left(p_left==0) = 0;
% log_right(p_right==0) = 0;
% ent_left = -p_left.*log_left - (1-p_left).*log(1-p_left);
% ent_right = -p_right.*log_right - (1-p_right).*log(1-p_right);
% ent_left(isnan(ent_left)) = 0;
% ent_right(isnan(ent_right)) = 0;
% 
% 
% gain_curve = (-size_left.*ent_left-size_right.*ent_right)/length(p);
% gain_curve(isnan(gain_curve)) = min(gain_curve(~isnan(gain_curve)));

end

