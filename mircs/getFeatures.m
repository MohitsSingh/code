function features = getFeatures(conf,trainingData)
for k = 1:length(trainingData)
    m = cropper(trainingData(k).img,round(trainingData(k).obj_rect));
    m = imResample(m,conf.features.winsize*8);
    face_rect = trainingData(k).face_rect;
    obj_rect = trainingData(k).obj_rect(1:4);
    features(k).X = conf.features.fun(m);
    features(k).G = geometric_features(face_rect,obj_rect);
    features(k).face_score = trainingData(k).img_data.faceScore;
    features(k).face_pose = trainingData(k).img_data.faceLandmarks.c;
end