function rois = getCandidateRoisBoxes(startRect,scaleFactor,samplingParams,I)
%%
%     params.sampling.nBoxThetas = 8;
% params.sampling.boxSize = [.3 .5 .7];

%%
d = 360/samplingParams.nBoxThetas;
thetas = 0:d:360-d;
ss = [sind(thetas(:)) cosd(thetas(:))];
prevSize = startRect(4)-startRect(2);
prevCenter = boxCenters(startRect);
radii = max(prevSize,scaleFactor*samplingParams.boxSize)*sqrt(2);
sizes = scaleFactor*samplingParams.boxSize;
p = 0;
rois = {};
% rois = struct('xy',{},'startPoint',{},'endPoint',{},'theta',{});
for iSize = 1:length(sizes)
    %curCenters = bsxfun(@plus,prevCenter,sizes(iSize)*ss/2);
    curCenters = bsxfun(@plus,prevCenter,radii(iSize)*ss/2);
    curBoxes = inflatebbox(curCenters,sizes(iSize),'both',true);
    for u = 1:size(curBoxes,1)
        bb = curBoxes(u,:);
        rois{end+1} = struct('xy',box2Pts(bb),...
            'startPoint',curCenters(u,:),'endPoint',curCenters(u,:),'theta',thetas(u));
    end
end