function rois = getCandidateRoisPolygons2(cur_config,scaleFactor,samplingParams,constrainAngle)

if nargin < 4
    constrainAngle = false;
end
thetas = samplingParams.thetas;
lengths = samplingParams.lengths*scaleFactor;
widths = samplingParams.widths*scaleFactor;

all_rois = {};
startPt = cur_config.endPoint;
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
if (constrainAngle)
    ad = 180*abs(angleDiff(pi*cellfun3(@(x) x.theta,rois)/180,pi*cur_config.theta/180))/pi;
    goods = ad <= samplingParams.maxThetaDiff;
    rois = rois(goods);
end

rois = cat(1,rois{:});
