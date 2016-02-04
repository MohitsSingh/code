function rois = getCandidateRoisPolygons(startPt,scaleFactor,samplingParams,onlyTheta,thetaToKeep)

if nargin < 5
    thetaToKeep = true(size(samplingParams.thetas));
end
thetas = samplingParams.thetas(thetaToKeep);
lengths = samplingParams.lengths*scaleFactor;
widths = samplingParams.widths*scaleFactor;
if onlyTheta
    lengths = max(lengths);
    widths = mean(widths);
end
all_rois = {};

for iLength = 1:length(lengths)
    for iWidth = 1:length(widths)
        curLength = lengths(iLength);
        curWidth = widths(iWidth);
        if curWidth > curLength
            continue
        end
        all_rois{end+1} = hingedSample(startPt,curWidth,curLength,thetas);
    end
end

rois = cat(2,all_rois{:});
