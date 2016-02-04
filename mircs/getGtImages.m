function objImages = getGtImages(conf,groundTruth,toRotate,inflation)
gtParts = {groundTruth.name};
% isObj = cellfun(@any,strfind(gtParts,objectName));
sourceImage = '';
objImages = {};
if (nargin < 4)
    inflation = 1;
end
for k = 1:length(groundTruth)
    k
    curGT = groundTruth(k);
    if (~isfield(curGT,'curTheta'))
        curGT.curTheta = 0;
    end
    %     if (isObj(k))
    if (~strcmp(curGT.sourceImage,sourceImage))
        I = getImage(conf,curGT.sourceImage);
        sourceImage = curGT.sourceImage;
    end
    z = false(dsize(I,1:2));
    if (~toRotate)
        curGT.curTheta = 0;
    end
    bbox = [ pts2Box([curGT.polygon.x,...
        curGT.polygon.y]) curGT.curTheta];    
    
    bbox(1:4) = round(bbox(1:4));        
    z(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
    z = false(dsize(I,1:2));
    bbox(1:4) = round(bbox(1:4));
    z(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
    z = imrotate(z,curGT.curTheta,'nearest','loose');
    [yy,xx] = find(z);
    I = imrotate(I,curGT.curTheta,'bilinear','loose');
    bbox = pts2Box([xx yy]);
    bbox = round(inflatebbox(bbox,inflation,'both'));
    objImages{k} = cropper(I,bbox);
    
end