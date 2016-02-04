function f = piotr_features2(x,sbin)
if (nargin == 0)
    
    x = zeros(80,80,3,'single');
    f = piotr_features(x,8);
    f = size(f,3);
    %     f = 42; % of course
    return;
end
if (nargin < 2)
    sbin = 8;
end

if (iscell(x))
    f = cellfun2(@(y) fhog(single(y),sbin),x);
    %f = cellfun2(@(PP) cat(3,cat(3,PP.data{1:3})/2,PP.data{4}), r);
else
    PP = chnsCompute(x,pChns);
    f = cat(3,cat(3,PP.data{1:3})/2,PP.data{4});
end

