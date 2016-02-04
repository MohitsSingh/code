function [res,face_boxes] = face_detection(im)
%FACE_DETECTION runs face detection in image.

curDir = pwd;
cd /home/amirro/code/3rdparty/voc-release5
for k = 1:10
    load(sprintf('models/face_big_%d_final.mat',k));
    models(k) = model;
end
global G_STARTUP; G_STARTUP = [];
startup;
res = struct('ds',{});
pyra = featpyramid(im, model);
for iModel = 1:length(models)
    iModel
    model = models(iModel);
    [ds, bs, trees] = gdetect(pyra, model, -1.5);
    top = nms(ds, 0.5);
    ds = ds(top,:);
    if (any(ds))
        ds(:,5) = iModel;
    end
    res(iModel).ds = ds;
end

ds = cat(1,res.ds);
top = nms(ds, 0.5);
face_boxes = ds(top,:);

cd(curDir);
end
