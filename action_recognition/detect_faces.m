function detections = detect_faces(I,model,resizeFactor,thresh)
if (nargin < 3)
    resizeFactor = 1;
end
if (nargin < 4)
    thresh = -2;
end
detections = struct('rot',{},'boxes',{});
I_orig = imResample(I,resizeFactor,'bilinear');
rots = 0;
for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');
    [ds, bs] = imgdetect(I, model,-2);
    top = nms(ds, 0.1);
    detections(iRot).rot = rots(iRot);
    if (isempty(top))
        detections(iRot).boxes= -inf(1,6);
    else
        detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
        detections(iRot).boxes(:,1:4) = detections(iRot).boxes(:,1:4)/resizeFactor;
    end
end