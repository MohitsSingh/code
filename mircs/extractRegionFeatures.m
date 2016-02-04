function [feats,labels,imageInds] = extractRegionFeatures(conf,regions,imgData,...
    toDebug_,minFaceScore,saveRegions)

% extract several types of features for region, including feature relating
% to the inter-relation between the different feature type.

conf.get_full_image = true;
learnParams = getDefaultLearningParams(conf);
featureExtractor = learnParams.featureExtractors{1};
% featureExtractor.bowConf = featureExtractor.bowConf(1);
for k = 1:length(featureExtractor.bowConf)
    featureExtractor.bowConf(k).bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf(k).bowmodel.numSpatialY = [1 2];
end

featureExtractor.doPostProcess = 0;
if (nargin < 5)
    minFaceScore = -1000;
end
if (nargin < 6)
    saveRegions = false;
end

feats = struct('face_occupancy',{},...
    'face_pose',{},...
    'face_area',{},...
    'mouth_area',{},...
    'face_score',{},...
    'face_color',{},...
    'region_occupancy',{},...
    'region_color',{},...
    'region_shape',{},...
    'mouth_occupancy',{},...
    'mouth_color',{},...
    'class',{},...
    'region_bof',{},...
    'imageIndex',{});
t = 0;
if (nargin < 4)
    toDebug_ = 0;
end
if (toDebug_)
    nFalse = 10;
    falseCount =0;
end

if (imgData.faceScore < minFaceScore)
    return;
end


% remove regions larger than half the image
N = numel(regions{1});
areas = cellfun(@nnz,regions);
regions((areas/N) > .5) = [];
if (isempty(regions))
    return;
end
regions = fillRegionGaps(regions);
[I,I_rect] = getImage(conf,imgData.imageID);
[xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imgData.faceLandmarks,...
    -I_rect);

face_mask = poly2mask2(face_poly,size(regions{1}));
mouth_mask = poly2mask2(mouth_poly,size(regions{1}));

% now, prune the regions which are not "interaction" regions.
[ovp_mouth,ints_mouth,~] = boxRegionOverlap(mouth_mask,regions);
[ovp_face,ints_face,~] = boxRegionOverlap(face_mask,regions);
[ovp_face_not,ints_not_face,areas] = boxRegionOverlap(~face_mask,regions);
% regions must intersect face,mouth and outside of face
must_have = ints_face > 0;
must_have = must_have & (ints_mouth > 0);
must_have = must_have & ints_not_face > 0;
regions = regions(must_have);


if (saveRegions && ~islogical(saveRegions))
    regions = regions(saveRegions);
end

[ovp_mouth,ints_mouth,~] = boxRegionOverlap(mouth_mask,regions);
[ovp_face,ints_face,~] = boxRegionOverlap(face_mask,regions);
[ovp_face_not,ints_not_face,areas] = boxRegionOverlap(~face_mask,regions);


landmarks_in_regions = findLandmarkOccupancy(regions,xy_c);

allRegions = [{face_mask};{mouth_mask};regions];

% extract features of "special" locations
face_score = imgData.faceScore;
face_pose = imgData.faceLandmarks.c;

[region_color,region_occupancy,edgeStrengths,region_shape,region_bow] = ...
    regionFeaturesHelper(conf,imgData.imageID,...
    allRegions,featureExtractor);

% add mouth features as well.

region_color = cat(2,region_color{:});
region_occupancy = cat(2,region_occupancy{:});
region_shape = [region_shape{:}];


face_color = region_color(:,1);

face_occ = region_occupancy(:,1);
mouth_color = region_color(:,2);
mouth_occ = region_occupancy(:,2);
face_bow = region_bow(:,1);
mouth_bow = region_bow(:,2);
[ignore1 ignore2 boxArea] = BoxSize(imgData.faceBox);
faceMaskArea = nnz(face_mask);
mouthMaskArea = nnz(mouth_mask);
for iRegion = 3:length(allRegions)
    t = t+1;
    feats(t).region = allRegions{iRegion};
    feats(t).ref_box_area = boxArea;
    feats(t).face_area = faceMaskArea;
    feats(t).mouth_area = mouthMaskArea;
    feats(t).face_score = face_score;
    feats(t).face_pose = face_pose;
    feats(t).face_occupancy = face_occ;
    feats(t).face_color = face_color;
    feats(t).face_bow = face_bow;
    feats(t).mouth_occupancy = mouth_occ;
    feats(t).mouth_color = mouth_color;
    feats(t).mouth_bow = mouth_bow;
    feats(t).region_color = region_color(:,iRegion);
    feats(t).region_occupancy = region_occupancy(:,iRegion);
    feats(t).region_area = region_shape(iRegion).Area;
    feats(t).region_bow = region_bow(:,iRegion);
    feats(t).MajorAxisLength = region_shape(iRegion).MajorAxisLength;
    feats(t).MinorAxisLength = region_shape(iRegion).MinorAxisLength;
    feats(t).Eccentricity = region_shape(iRegion).Eccentricity;
    feats(t).Orientation = region_shape(iRegion).Orientation;
    feats(t).edgeStrength = edgeStrengths(:,iRegion);
    feats(t).ovp_mouth = ovp_mouth(iRegion-2);
    feats(t).int_region_mouth = ints_mouth(iRegion-2);
    feats(t).ovp_region_face = ovp_face(iRegion-2);
    feats(t).ints_region_face = ints_face(iRegion-2);
    feats(t).ovp_region_face_not = ovp_face_not(iRegion-2);
    feats(t).ints_region_not_face = ints_not_face(iRegion-2);
    feats(t).landmark_occ = landmarks_in_regions(:,iRegion-2);
    feats(t).regionID = iRegion-2; % for debugging
    if (saveRegions)
        %             if (isLogical(saveRegions) || saveRegions == iRegion-2)
        feats(t).region = regions{iRegion-2}; % warning: for debugging/visualization only- very memory consuming!
        %             end
    end
end
labels = [feats.class];

function [colors,occupancies,edgeStrengths,shapes,region_bow] = regionFeaturesHelper(conf,imageID,regions,...
    featureExtractor)
[I] = getImage(conf,imageID);
region_bow = featureExtractor.extractFeatures(imageID,regions);
% region_bow = zeros(size(regions));
n = length(regions);
occupancies = cellfun2(@(x)col(getOccupancy(x)),regions);
colors = cellfun2(@(x) col(meanSegmentColor(I,x)),regions);
shapes = cellfun2(@segmentShape,regions);
[ucm,gpb] = loadUCM(conf,imageID);
q = addBorder(zeros(size(ucm)),1,1);
ucmStrengths = cellfun(@(x) mean(ucm(imdilate(bwperim(x),ones(3)) & (ucm>0 & ~q))),regions);
gpbStrengths = cellfun(@(x) mean(gpb(imdilate(bwperim(x),ones(3)) & (ucm>0 & ~q))),regions);
M = gradientMag(im2single(I),1);
gradientStrengths = cellfun(@(x) mean(M(imdilate(bwperim(x),ones(3)) & (ucm>0 & ~q))),regions);

edgeStrengths = [ucmStrengths,gpbStrengths,gradientStrengths]';


% feature functions
function x = getOccupancy(x)
x =  imResample(single(x),[5 5]);
x = x>0;
% x = x/sum(x(:)); % L1 normalize
%x = x/sum(x(:).^2).^.5;
function c = meanSegmentColor(I,seg)
c = zeros(3,1);
for k = 1:3
    I_ = I(:,:,k);
    c(k) = mean(I_(seg));
end
function shape = segmentShape(x)
shape = regionprops(x,'Eccentricity','Orientation','Area','MajorAxisLength','MinorAxisLength');
shape = shape(1);
shape.Orientation = [cosd(shape.Orientation) sind(shape.Orientation)];



%function landmarks_in_regions = findLandmarkOccupancy(regions,landmarks,ref_box)
function landmarks_in_regions = findLandmarkOccupancy(regions,bc)
frontal_n = 68;
profile_n = 39;

% xy = landmarks.xy;
% bc = round(bsxfun(@minus,boxCenters(xy),ref_box([1 2])));
inds = sub2ind2(size(regions{1}),round(fliplr(bc)));
landmarks_in_regions = cellfun2(@(x) x(inds),regions);
landmarks_in_regions = cat(2,landmarks_in_regions{:});
assert (size(bc,1) == frontal_n || size(bc,1) == profile_n);
if (size(bc,1) == frontal_n)
    landmarks_in_regions = [landmarks_in_regions;zeros(profile_n,length(regions))];
else
    landmarks_in_regions = [zeros(frontal_n,length(regions));landmarks_in_regions];
end


