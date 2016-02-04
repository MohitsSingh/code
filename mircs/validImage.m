function t = validImage(imageSet,k,posOnly,minFaceScore)
t = true;
if (~imageSet.labels(k) && posOnly)
    t = false;
end
if (imageSet.faceScores(k) < minFaceScore)
    t = false;
end

end