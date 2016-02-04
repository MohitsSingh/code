function faceData = getFaceData(imgData)
faceData.faceScore = imgData.faceScore;
faceData.facePose = imgData.faceLandmarks.c;
faceData.faceBox = imgData.faceBox;
end