function imgData = switchToGroundTruth(imgData)
if (imgData.isTrain && imgData.indInFraDB~=-1)
    imgData.faceBox = imgData.faceBox_gt;
    imgData.mouth = imgData.mouth_gt;
    imgData.objects = imgData.objects_gt;
end