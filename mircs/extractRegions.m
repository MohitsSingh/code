function [regions_sub,groups,bbox,face_mask,mouth_mask,I,params] = extractRegions(conf,imageSet,k,params)
if (nargin < 4)
    params.inflationFactor = 1.1;
    params.regionExpansion = 2;
    params.ucmThresh = .15;
end

regions_sub = [];
groups = [];
bbox = [];
face_mask = [];
mouth_mask = [];
I = [];

conf.get_full_image = false;
currentID = imageSet.imageIDs{k};
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);

ucm = loadUCM(conf,currentID);
ucm = ucm(ymin:ymax,xmin:xmax); 
bbox = round(imageSet.faceBoxes(k,1:4));
bbox = round(inflatebbox(bbox,params.inflationFactor*[1 1],'both',false));
I = cropper(I,bbox);
[a1 a2] = BoxSize(bbox);
if (min(a1,a2) < 10)
    return;
end
subUCM = cropper(ucm,bbox);
faceLandmarks = imageSet.faceLandmarks(k);

if (~isfield(faceLandmarks,'face_outline'))
    xy = faceLandmarks.xy;
    if (isempty(xy))
        return;
    end
    xy_c = boxCenters(xy);
    if (size(xy_c,1)==68)
        outline_ = [68:-1:61 52:60 16:20 31:-1:27];
    else
        outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
    end
    
    c_boxes = xy(outline_,:);
    c_boxes = c_boxes-repmat(bbox(1:2),size(c_boxes,1),2);
    
    c_poly = boxesToEdges(c_boxes,subUCM);
    chull = convhull(c_poly(:,1),c_poly(:,2));
    c_poly = c_poly(chull,:);
%         figure,imagesc(I);
%         hold on;
%         plotBoxes(c_boxes,'g');
%         plot(c_poly(:,1),c_poly(:,2),'r.');
%         figure,imagesc(subUCM);
    
    %     c_poly = xy_c(outline_,:);
    %     c_poly = bsxfun(@minus,c_poly,bbox(1:2));
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(subUCM,1),size(subUCM,2));
    % remove artefacts caused by moving points around...
    rprops = regionprops(face_mask,'area','PixelIdxList');
    [~,ia] = max([rprops.Area]);
    face_mask = false(size(face_mask));
    face_mask(rprops(ia).PixelIdxList) = true;
    
    %     face_mask = snapToEdges(face_mask,subUCM);
    %face_mask = approximateRegion(face_mask,regions_sub,4);
    face_mask = imdilate(face_mask,ones(5));
    
    
    mouthBox = round(imageSet.lipBoxes(k,1:4));
    mouthBox = mouthBox-bbox([1 2 1 2]);
else
    face_mask = poly2mask(faceLandmarks.face_outline(:,1),faceLandmarks.face_outline(:,2),...
        size(subUCM,1),size(subUCM,2));
    %     face_mask = faceLandmarks.face_seg;
    %     face_mask = cropper(face_mask,bbox);
    %     face_mask = imclose(face_mask,ones(3));
    % next line is in case of bug where face mask is slightly smaller than
    % the face image.
    %     face_mask = padarray(face_mask,max(0,dsize(E,1:2)-size(face_mask)),0,'post');
    %     face_mask = face_mask(1:size(E,1),1:size(E,2));
    mouthBox = round(imageSet.lipBoxes(k,1:4));
    %      mouthBox = round(inflatebbox(mouthBox,[4 3],'both',false));
    %     face_mask = face_mask(bbox(2):bbox(4),bbox(1):bbox(3));
end

mouth_mask = false(size(subUCM));
mouthBox = clip_to_image(mouthBox,subUCM);
mouth_mask(mouthBox(2):mouthBox(4),mouthBox(1):mouthBox(3)) = true;
mouth_mask = mouth_mask & face_mask; % don't allow mouth to be out of face :-)
regions_sub = combine_regions_new(subUCM,params.ucmThresh);
[groups] = expandRegions(regions_sub,params.regionExpansion);
