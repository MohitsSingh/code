function [rr] = hingedSample(startPt,avgWidth,avgLength,thetas)
if nargin < 4
%    thetas = 0:20:360;   
end
rr = {};
for iTheta = 1:length(thetas)
    rr{iTheta} = directionalROI_rect(startPt,thetas(iTheta),avgWidth,avgLength);
end

% rr = cat(1,rr{:});

