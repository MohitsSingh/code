function detections = occludeAndExtract(I,param,w)
layers = param.layers;
net = param.net;
meanImage = param.net.normalization.averageImage;
resizeRatio = size(I,1)/param.net.normalization.imageSize(1);
I = imResample(double(im2uint8(I)), param.net.normalization.imageSize(1:2));
wndSize = round(size(I,1)*param.objToFaceRatio);
j = max(1,round(wndSize/2));
d = 1:j:size(I,1)-wndSize+1;
[xx,yy] = meshgrid(d,d);
n = length(xx(:));
detections = zeros(n,5);
detections(:,1:4) = [xx(:),yy(:),xx(:)+wndSize,yy(:)+wndSize];
detections(:,5) = -inf;
z = cell(n,1);
for u = 1:n
    r = I;
    r(yy(u):yy(u)+wndSize,xx(u):xx(u)+wndSize,:) = meanImage(yy(u):yy(u)+wndSize,xx(u):xx(u)+wndSize,:);
    z{u} = uint8(r);
%     clf; imagesc2(r/255); dpc;
end
%windows = multiCrop2(I,detections);
feats = extractDNNFeats(z,net,layers,false);
feats = feats.x;
detections(:,5) = w'*feats;
detections(:,1:4) = resizeRatio*detections(:,1:4);