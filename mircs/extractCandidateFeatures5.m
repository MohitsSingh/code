function [feats,regions,scores,bbox] = extractCandidateFeatures5(conf,imageSet,salData,k,debug_,debug_fun,qq)
%Rs = extractCandidateFeatures3(conf,imageSet,salData,k)
feats = [];
if (nargin < 5)
    debug_ = false;
else
    if (nargin < 6)
        debug_fun = @(x) x.ucmStrengths-x.ints_face+x.ints_mouth;
    end
end


% some parameters:

inflationFactor = 1;
regionExpansion = 2;
ucmThresh = .15;

currentID = imageSet.imageIDs{k};
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);

ucmFile = fullfile(conf.gpbDir,strrep(currentID,'.jpg','_ucm.mat'));
gpbFile = fullfile(conf.gpbDir,strrep(currentID,'.jpg','.mat'));
L_gpb = load(gpbFile);
load(ucmFile); % ucm
ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
gpb = L_gpb.gPb_thin(ymin:ymax,xmin:xmax);
bbox = round(imageSet.faceBoxes(k,1:4));
bbox = round(inflatebbox(bbox,inflationFactor*[1 1],'both',false));
I = cropper(I,bbox);

if (min(dsize(I,1:2)) < 10)
    %     clf; imagesc(I); axis image; drawnow
    return;
end
subUCM = cropper(ucm,bbox);
% mexSEEDS
subGPB = cropper(gpb,bbox);
if (isempty(subUCM))
    return;
end
E = subUCM;
[M,O] = gradientMag(im2single(I),1);
if (debug_)
    figure(1);clf;
    subplot(1,2,1); imagesc(I); axis image;
    subplot(1,2,2); imagesc(M); colorbar; axis image;
end
% pause;
% regions = getRegions(conf,currentID,false);




% segments = vl_slic(single(vl_xyz2luv(vl_rgb2xyz(im2single(I)))), 10, 1);
% segments = vl_slic(single(vl_xyz2luv(vl_rgb2xyz(im2single(I)))), 10,.0001);
% segments = vl_slic(single(vl_xyz2luv(vl_rgb2xyz(im2single(I)))), 20,.0001);
% segImage  = paintSeg(I,segments);
%
% imagesc(segImage);


faceLandmarks = imageSet.faceLandmarks(k);

if (~isfield(faceLandmarks,'face_outline'))
    xy = faceLandmarks.xy;
    if (isempty(xy))
        return;
    end
    xy_c = boxCenters(xy);
    if (size(xy_c,1)==68)
        outline_ = [68:-1:61 52:60 16:20 31:-1:27];
        inner_lips = [36 37 38 42 42 45 47 49 36];
        outer_lips = [35 34 33 32 39 40 41 44 46 51 48 50 35];
        mouth_corner = [35 41];
    else
        outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
        inner_lips = [25 24 26 23 27];
        outer_lips = [16:22];
        mouth_corner = 19;
    end
    
    %     chull = 1:size(xy_c);
    % find the occluder!! :-)
    
    c_boxes = xy(outline_,:);
    c_boxes = c_boxes-repmat(bbox(1:2),size(c_boxes,1),2);
    c_poly = boxCenters(c_boxes);
    %     c_poly = boxesToEdges(c_boxes,subUCM);
    %     chull = convhull(c_poly(:,1),c_poly(:,2));
    %     c_poly = c_poly(chull,:);
    %     figure,imagesc(I);
    %     hold on;
    %     plotBoxes(c_boxes,'g');
    %     plot(c_poly(:,1),c_poly(:,2),'r.');
    %     figure,imagesc(subUCM);
    
    %     c_poly = xy_c(outline_,:);
    %     c_poly = bsxfun(@minus,c_poly,bbox(1:2));
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));
    % remove artefacts caused by moving points around...
    rprops = regionprops(face_mask,'area','PixelIdxList');
    [~,ia] = max([rprops.Area]);
    face_mask = false(size(face_mask));
    face_mask(rprops(ia).PixelIdxList) = true;
    
    %     face_mask = snapToEdges(face_mask,subUCM);
    %face_mask = approximateRegion(face_mask,regions_sub,4);
    %     face_mask = imdilate(face_mask,ones(5));
    
    
    mouthBox = round(imageSet.lipBoxes(k,1:4));
    mouthBox = mouthBox-bbox([1 2 1 2]);
else
    face_mask = poly2mask(faceLandmarks.face_outline(:,1),faceLandmarks.face_outline(:,2),...
        size(E,1),size(E,2));
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

mouth_mask = false(size(E));
mouthBox = clip_to_image(mouthBox,E);
mouth_mask(mouthBox(2):mouthBox(4),mouthBox(1):mouthBox(3)) = true;
mouth_mask = mouth_mask & face_mask; % don't allow mouth to be out of face :-)
% if (debug_)
% %     figure(3);clf; imagesc(mouth_mask);
% end
face_boundary = bwperim(face_mask);
mouth_boundary = bwperim(mouth_mask);

[ii,jj,vv] = find(subUCM);
if (length(ii) < 5)
    return;
end

E = E>.2;
if (~any(E(:)))
    return
end
E = bwmorph(E,'thin',Inf);
if (nnz(E) < 5)
    return
end

if (debug_)
end

regions_sub = combine_regions_new(subUCM,ucmThresh);
[~,regions_sub]= expandRegions(regions_sub,regionExpansion);

regions_sub = fillRegionGaps(regions_sub);

areas = cellfun(@nnz,regions_sub);
regions_sub((areas/numel(E))>.5) = [];
if (isempty(regions_sub))
    return;
end
areas(areas/numel(E)>.5) = [];

q = addBorder(zeros(size(E)),1,1);

ucmStrengths = cellfun(@(x) mean(subUCM(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);
gpbStrengths = cellfun(@(x) mean(subGPB(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);
gradientStrengths = cellfun(@(x) mean(M(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);

M = M .* ~imdilate(face_boundary,ones(3));
subUCM = subUCM .* ~imdilate(face_boundary,ones(3));
subGPB = subGPB .* ~imdilate(face_boundary,ones(3));

ucmStrengths_noface = cellfun(@(x) mean(subUCM(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);
gpbStrengths_noface = cellfun(@(x) mean(subGPB(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);
gradientStrengths_noface = cellfun(@(x) mean(M(imdilate(bwperim(x),ones(3)) & (subUCM>0 & ~q))),regions_sub);


% now find some properties on the regions.

[ovp_mouth,ints_mouth,~] = boxRegionOverlap(mouth_mask,regions_sub);
[ovp_face,ints_face,~] = boxRegionOverlap(face_mask,regions_sub);
[ovp_face_not,ints_not_face,areas] = boxRegionOverlap(~face_mask,regions_sub);
% region must: intersect with face and mouth, and outside of face too.
% must_have = r.ints_face > 0;
% must_have = must_have & (r.ints_mouth > 0);
% must_have = must_have & ints_not > 0;


bboxes = zeros(length(regions_sub),4);
for k = 1:length(regions_sub)
    [y,x] = find(regions_sub{k});
    bboxes(k,:) = pts2Box([x,y]);
    bboxes(k,[1 3]) = bboxes(k,[1 3])/size(E,2);
    bboxes(k,[2 4]) = bboxes(k,[2 4])/size(E,1);
    %bboxes(k,:) = pts2Box([x,y])/size(E,1);
end
% feats.regions = regions_sub;
%feats.areas = areas/numel(E);

feats.ovp_mouth = ovp_mouth; % ovp of mouth with regions
feats.ints_mouth = ints_mouth; % intesection of regions with mouth
feats.ovp_face = ovp_face;
feats.ints_face = ints_face;
feats.ints_not = ints_not_face;
feats.ucmStrengths = ucmStrengths;
feats.gradientStrengths = gradientStrengths;
feats.gpbStrengths = gpbStrengths;
feats.gpbStrengths_noface = gpbStrengths_noface;
feats.ucmStrengths_noface = ucmStrengths_noface;
feats.gradientStrengths_noface = gradientStrengths_noface;

feats.bboxes = bboxes;
feats.winSize = size(E);
feats.areas = areas;
feats.face_area = nnz(face_mask);
feats.mouth_area = nnz(mouth_mask);
scores = feval(debug_fun,feats);

regions = regions_sub;
if (debug_)
    figure(2);clf;
    subplot(1,3,1); imagesc(I);    axis image;
    subplot(1,3,2); imagesc(I);    axis image;
    hold on;
    
    xy = bwboundaries(mouth_mask);
    xy = fliplr(xy{1});
    plot(xy(:,1),xy(:,2),'g','LineWidth',2);
    %hold on; plotBoxes2(mouthBox([2 1 4 3]),'g','LineWidth',2);
    xy = bwboundaries(face_mask);
    xy = fliplr(xy{1});
    plot(xy(:,1),xy(:,2),'m--','LineWidth',2);
    %subplot(2,2,2); imagesc(face_mask+mouth_mask);axis image;
    
    curScores = feval(debug_fun,feats);
    [rr,irr] = sort(curScores,'descend');
    
    %imagesc(regions_sub{ii} & ~face_mask);
    
    ii = irr(1);
    nnz(curScores)
    %     feats.gradientStrengths(ii)
    
    subplot(1,3,3); displayRegions(I,regions_sub,curScores,0,1);
    %     saveas(gcf,sprintf('/home/amirro/mircs/experiments/experiment_0013/%04.0f.png',qq));
    %ucmStrengths+-ints_face+ints_mouth);
    return;
end