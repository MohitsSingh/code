function [ bbs ] = uv2boxes( conf,uu,vv,scales,t,winSize)
%UV2PIX Summary of this function goes here
%   Detailed explanation goes here
if (isempty(uu))
    bbs = [];
    return;
end
o = [uu(:) vv(:)] - t.padder;
ws = winSize;
if (isscalar(conf))
    sbin = conf;
else
    sbin = conf.detection.params.init_params.sbin;
end
bbs = ([o(:,2) o(:,1) o(:,2)+ws(2) ...
    o(:,1)+ws(1)] - 1) .* ...
    repmat(sbin./scales',1,4) + 1 + repmat([0 0 -1 ...
    -1],length(scales),1);
bbs(:,5:12) = 0;
bbs(:,5) = (1:size(bbs,1));
bbs(:,6) = 0;
bbs(:,8) = scales;
bbs(:,9) = uu;
bbs(:,10) = vv;