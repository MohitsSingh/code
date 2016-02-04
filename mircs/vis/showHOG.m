function hogPic = showHOG( conf,w )
%SHOWHOG Summary of this function goes here
%   Detailed explanation goes here
    if (isempty(conf))
        conf.features.winsize = size(w);
    end
    if (isstruct(w))
        w = w.w;
    end
    
    hogPic = jettify(HOGpicture(reshape(w,conf.features.winsize(1),conf.features.winsize(2),[]),15));
end