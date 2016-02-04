function [patches,polys,angles] = get_pos_samples_restricted(conf,imgData,param,gtDir,toResize,gt_mouth_corners)
patches = cell(1,length(imgData));
polys = cell(size(imgData));
angles = zeros(size(imgData));
if (nargin < 5)
    toResize = false;
end
scaleRatio = param.objToFaceRatio;
for t = 1:length(imgData)
    curPath = getImagePath(conf,imgData(t).imageID);
    [cur_poly,curAngle] = loadGt(curPath,gtDir);
    polys{t} = cur_poly;
    angles(t) = curAngle;
    I = getImage(conf,imgData(t));
    %gt_mouth_corners{t}
    
    curCenter = mean(gt_mouth_corners{t});
    faceBox = imgData(t).faceBox;
    face_scale = faceBox(3)-faceBox(1);
    mouth_bb = round(inflatebbox(curCenter,scaleRatio*face_scale,'both',true));
    poly_i = intersect_poly_bb(cur_poly,mouth_bb);
    
        
%     x2(I); plotPolygons(gt_mouth_corners{t},'r+');    
%     plotBoxes(bb);
%     plotPolygons(cur_poly,'g-');
%     plotPolygons(poly_i,'m-');
%     
    mouth_bb = round(pts2Box(poly_i));
    
%     [x3, y3] = polybool(operation, x1, y1, x2, y2, varargin)
    
    curPatch = im2single(cropper(I,mouth_bb));
    if toResize
        curPatch = imResample(curPatch,param.windowSize,'bilinear');
    end
    patches{t} = curPatch;
end

function p = intersect_poly_bb(poly,bb)
    x1 = poly(:,1); y1 = poly(:,2);
    [x1,y1] = poly2cw(x1,y1);
    bb_poly = box2Pts(bb);
    x2 = bb_poly(:,1); y2 = bb_poly(:,2);
    [x2,y2] = poly2cw(x2,y2);
    [x3,y3] = polybool('&',x1,y1,x2,y2);
    p = [x3 y3];
    