
function res = log_lh( obj,X )
%LOG_LH get log-likelyhood of 2d gaussian distribution
covNames = { 'diagonal','full'};
CovType = find(strncmpi(obj.CovType,covNames,length(obj.CovType)));
res = my_wdensity(X,obj.mu, obj.Sigma, obj.PComponents, obj.SharedCov, CovType);
%   Detailed explanation goes here
end

