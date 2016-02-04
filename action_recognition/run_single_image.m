function res = run_single_image(conf,imagePath)
I = imread(fullfile(conf.baseDir,imagePath));
%
rescaleFactor = 2;
res.faces = detect_faces(I,conf.face_det_model,rescaleFactor,-1);
face_images = {};
bbox_inflation_factor = 1.5;
for t = 1:size(res.faces.boxes,1)
    curBox = res.faces.boxes(t,1:4);
    if (isinf(curBox(1)))
        continue
    end
    curBox = inflatebbox(curBox,bbox_inflation_factor,'both',false);
    face_images{t} = cropper(I, round(curBox));
end
res.face_images = face_images;
res.imagePath = imagePath;
res.deep_nn_global_feats =extractDNNFeats(I,conf.net);
if (~isempty(face_images))
    res.deep_nn_face_feats = extractDNNFeats(face_images,conf.net);
end