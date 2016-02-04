function res = getOcclusionData(conf,imgData,roi,regions)
if (nargin < 3 || isempty(roi))
    roi = [];
end

res.occlusionPatterns = [];
res.dataMatrix = [];
res.regions = [];
if (nargin < 4)
    [occlusionPatterns,regions,face_mask,mouth_mask] = getOcclusionPattern(conf,imgData,...
        'roi',roi);
else
    [occlusionPatterns,regions,face_mask,mouth_mask] = getOcclusionPattern(conf,imgData,...
        'roi',roi,'regions',regions);
end
res.occlusionPatterns = occlusionPatterns;
res.face_mask = face_mask;
res.mouth_mask = mouth_mask;
res.regions = regions;
end