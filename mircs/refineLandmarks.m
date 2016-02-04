function [res,faceImages] = refineLandmarks(conf,ids,landmarks)
    [~,~,faceBoxes] = landmarks2struct(landmarks,ids,conf);
    faceBoxes = inflatebbox(faceBoxes,1.3);
    faceImages = multiCrop(conf,ids,round(faceBoxes/2));            
    res = detect_landmarks(conf,faceImages,2);   
end