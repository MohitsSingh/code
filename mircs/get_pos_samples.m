function [patches,polys,angles] = get_pos_samples(conf,imgData,param,gtDir,toResize)
patches = cell(1,length(imgData));
polys = cell(size(imgData));
angles = zeros(size(imgData));
if (nargin < 5)
    toResize = false;
end
scaleRatio = param.objToFaceRatio;
for t = 1:length(imgData)
    %     curPath = imgData(t).image_path;
    curPath = getImagePath(conf,imgData(t).imageID);
    [cur_poly,curAngle] = loadGt(curPath,gtDir);
    polys{t} = cur_poly;
    angles(t) = curAngle;
    %     cur_poly = imgData(t).gt_obj;
    %if iscell(curPath),curPath = curPath{1};end;
    I = getImage(conf,imgData(t));
    %I = imread2(curPath);
    %     curAngle = imgData(t).gt_angle;
    cur_poly = rotate_pts(cur_poly,pi*curAngle/180,fliplr(size2(I))/2);
    I = imrotate(I,curAngle,'bilinear','crop');
    bb = makeSquare(pts2Box(cur_poly));
    curBox = imgData(t).faceBox;
    face_scale = curBox(3)-curBox(1);
    bb = round(inflatebbox(bb,scaleRatio*face_scale,'both',true));
    curPatch = im2single(cropper(I,bb));
    if toResize
        curPatch = imResample(curPatch,param.windowSize,'bilinear');
    end
    patches{t} = curPatch;
end