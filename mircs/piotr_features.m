function f = piotr_features(x,sbin)
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

pChns = chnsCompute();
pChns.shrink = sbin;
pChns.pGradMag.colorChn = 1;
pChns.pGradHist.nOrients = 6;
pChns.pCustom(1).enabled = 1;
pChns.pCustom.name = 'fhog';
pChns.pCustom.hFunc = @fhog;
pChns.pCustom.pFunc = {};


if (iscell(x))
    r = cellfun2(@(y) chnsCompute(y,pChns),x);
    f = cellfun2(@(PP) cat(3,cat(3,PP.data{1:3})/2,PP.data{4}), r);
else
    PP = chnsCompute(x,pChns);
    f = cat(3,cat(3,PP.data{1:3})/2,PP.data{4});
end

