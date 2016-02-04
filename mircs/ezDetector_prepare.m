function [pos_samples,neg_samples] = ezDetector_prepare(pos_images,u,param) 
    layers = param.layers;
    %layers = 17; % fc6? 
    net = param.net;
    [res] = extractDNNFeats(pos_images,net,layers,false)
    pos_samples = res.x;
    
    % get some negative sub-windows
    
    %randi([1 size(u,1)-    
    
    wndSize = round(size(u{1},1)*param.objToFaceRatio);
    j = round(wndSize/2);
    d = 1:j:size(u{1},1)-wndSize;
    [xx,yy] = meshgrid(d,d);
    n = length(xx(:));

    negs = {};
    for t = 1:length(u)
        p = randi(n,[1 1]);
        boxes = [xx(p) yy(p),[xx(p) yy(p)]+wndSize];
        curPatches = multiCrop2(u{t},boxes);
        %curBox = [xx(p) yy(p),[xx(p) yy(p)]+wndSize];
        negs{end+1} = curPatches;
        %cropper(u{t},curBox);
    end
    negs = cat(2,negs{:});
    [res] = extractDNNFeats(negs,net,layers,false);
    neg_samples = res.x;
    
    
    
    %[res,rects] = extractDNNFeats_tiled(imgs,net,tiles,layers,prepareSimple)
end
