%function [detections,feats] = ezDetector(w,I,param,restrictRule,restrictConfig)
function [detections,feats] = ezDetector(w,I,param,varargin)



ip = inputParser;
ip.addOptional('restrictRule',[]);
ip.addOptional('restrictConfig',[]);
ip.addOptional('anchorBoxes',[]);
ip.parse(varargin{:});
restrictRule = ip.Results.restrictRule;
restrictConfig = ip.Results.restrictConfig;
anchorBoxes = ip.Results.anchorBoxes;

layers = param.layers;
net = param.net;
if isempty(restrictRule)
    restrictRule = @(x,y) true(size(x,1),1);
end
%     if (nargin < 4)
%         restriction_bb = [1 1 size(I,2) size(I,1)];
%         restrictRule = @(x,y) true(size(x,1),1);
%     end

if isempty(anchorBoxes)
%     detections = zeros(size(anchorBoxes,1),1,5);
% else
    
    wndSize = round(size(I,1)*param.objToFaceRatio);
    
    %     j = max(1,round(wndSize/16));
    j =2;
    d = 1:j:size(I,1)-wndSize;
    [xx,yy] = meshgrid(d,d);
    
    n = length(xx(:));
    detections = zeros(n,5);
    detections(:,1:4) = [xx(:),yy(:),xx(:)+wndSize,yy(:)+wndSize];
    
    goods = restrictRule(detections,restrictConfig);
    %     [ overlaps ,ints] = boxesOverlap( detections(:,1:4),restriction_bb);
    %     [~,~,areas] = BoxSize(detections(:,1:4));
    %     goods = ints./areas > 0;
    %    detections(:,5) = -inf;
    detections = detections(goods,:);
else
    detections = anchorBoxes;
end
windows = multiCrop2(I,detections);
%     for t = 1:n
%
%         curBox = [xx(t) yy(t),[xx(t) yy(t)]+wndSize];
%         detections(t,1:4) = curBox;
%         windows{t} = cropper(I,curBox);
%     end
feats = extractDNNFeats(windows,net,layers,false);
feats = feats.x;
%detections(goods,5) = w'*feats;
detections(:,5) = w'*feats;
%[res,rects] = extractDNNFeats_tiled(imgs,net,tiles,layers,prepareSimple)
end
