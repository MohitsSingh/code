function objImages = markData(conf,groundTruth,objectName)
gtParts = {groundTruth.name};
isObj = cellfun(@any,strfind(gtParts,objectName));
sourceImage = '';
objImages = {};
for k = 1:length(groundTruth)
    curGT = groundTruth(k);
    if (isObj(k))
        if (~strcmp(curGT.sourceImage,sourceImage))
            I = getImage(conf,curGT.sourceImage);
            sourceImage = curGT.sourceImage;
        end
        z = false(dsize(I,1:2));
        bbox = [ pts2Box([curGT.polygon.x,...
            curGT.polygon.y]) curGT.Orientation];
        bbox(1:4) = round(bbox(1:4));
        z(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
        objImages{end+1} =  I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    end
end