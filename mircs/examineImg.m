function examineImg(conf,imageID,img_db)
if (nargin == 2)
    imgData = imageID;
else
    imgData = img_db(findImageIndex(img_db,imageID));
end

I_orig = getImage(conf,imageID);
x2(I_orig);
plotBoxes(imgData.faceBox);
end