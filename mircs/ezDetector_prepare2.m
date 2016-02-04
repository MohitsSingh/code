function [pos_samples,neg_samples] = ezDetector_prepare2(pos_images,neg_images,param) 
    layers = param.layers;
    %layers = 17; % fc6? 
    net = param.net;
    [res] = extractDNNFeats(pos_images,net,layers,false)
    pos_samples = res.x;       
    [res] = extractDNNFeats(neg_images,net,layers,false);
    neg_samples = res.x;
    
    
    
    %[res,rects] = extractDNNFeats_tiled(imgs,net,tiles,layers,prepareSimple)
end
