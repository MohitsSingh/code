function feats = extractFeaturesForSegment(img,mask,mouthMask,featureExtractor,debug_,params)
% extract multiple features from the image given the mask (both appearance
% and shape features)

% initiate stuff for feature extractors

% 1. "global" mask features
if (nargin < 4)
    debug_ = false;
end

if (debug_)
    img_orig = img;
end
shapeFeats = {};
shapeFeatNames = {};
appearanceFeats = {};
appearanceFeatNames = {};

mouthBox = box2Pts(mouthMask);


% 
% mouthBox = inflatebbox([30 30 30 30],[30 10],'both','true');
% mouthBox_larger = inflatebbox([30 30 30 30],[30 30],'both','true');

% clf; imagesc2(img); plotBoxes(mouthBox_larger); pause;

% count the occupancy of the mask in each of the 9 mouth regions,
% which are the box itself and its 8 neighbors.
dx = 30;
dy = 10;
boxes_n = zeros(8,4);
t = 0;
for x = -dx:dx:dx
    for y = -dy:dy:dy
        t = t+1;
        boxes_n(t,:) = mouthBox + [x y x y];
    end
end
boxes_n = clip_to_image(boxes_n,mask);
boxSums = sum_boxes(double(mask),boxes_n);
shapeFeats{end+1} = double(boxSums>0);
shapeFeatNames{end+1} = 'binary occupancy around mouth';

% co-occurence matrix.
U = double(boxSums>0)*double(boxSums>0)';
U = U(~eye(size(U,1)));
shapeFeats{end+1} = U;
shapeFeatNames{end+1} = 'occupancy around mouth - pairs';

% class specific features
boxSums = reshape(boxSums,3,3);

%computeHeatMap(img,[boxes_n [1:9]'],'max');

occupancyMask = imResample(double(mask),[9 9],'bilinear');
occupancyMaskCenter = cropper(mask,mouthBox);
% unrotated appearance features
% featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
% featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
appearanceFeats{end+1} = featureExtractor.extractFeatures(imresize(img,2,'bilinear'),[],'normalization','Improved');
appearanceFeatNames{end+1} = 'global fisher appearance';
% featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
% featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
appearanceFeats{end+1} = featureExtractor.extractFeatures(imresize(img,2,'bilinear'),...
    imresize(poly2mask2(box2Pts(mouthBox_larger),size2(img)),2,'nearest'),'normalization','Improved');
appearanceFeatNames{end+1} = 'mouth fisher appearance';

curMask = imresize(mask,2,'nearest');
squarifyRegion = true; %TODO!!! squarifying the mask for debugging purposes
if (squarifyRegion)
    regionBox = region2Box(curMask);
    %regionBox = inflatebbox(regionBox,wSize*[1 1],'both',true);
    curMask = poly2mask2(box2Pts(regionBox),size2(curMask));
end
appearanceFeats{end+1} = featureExtractor.extractFeatures(imresize(img,2,'bilinear'),curMask,'normalization','Improved');
appearanceFeatNames{end+1} = 'object fisher appearance';
occupancyMask_binary = imResample(double(mask),[9 9],'nearest');
shapeFeats{end+1} = occupancyMask;
shapeFeatNames{end+1} = 'global occupancy mask';
shapeFeats{end+1} = occupancyMaskCenter;
shapeFeatNames{end+1} = 'mouth-vicinity occupancy mask';

[logPolarMask,m_vis] = getLogPolarShape(mask,36);
shapeFeats{end+1} = logPolarMask;
shapeFeatNames{end+1} = 'log polar shape';
r = regionprops(mask,'Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation','Solidity');
r = r(1);

% 2. canonized mask features.

% [y,x] = find(occupancyMask);
x = zeros(size(occupancyMask));
x(5,5) = 1;
x = bwdist(x);x = 1./(1+5*x);
[y,x,v] = find(occupancyMask.*x);
mm = sum([x.*v y.*v],1)/sum(v);
mm_ = mm;
mm = mm-fliplr(size2(occupancyMask))/2;

% check if a thin object
before = nnz(occupancyMask_binary);
after = nnz(imopen( occupancyMask_binary,ones(3)));
%disp(before - after)
isThin = (before-after)./before > .5;
if (~all(mm==0))
    mm = normalizeVector(mm);
end
if isThin && (r.Eccentricity > .85 &&  r.Solidity > .8 || r.Eccentricity > .9)
    theta = r.Orientation-90;
else
    theta = atan2d(mm(1),mm(2));
end

shapeFeats{end+1} = [r.Area r.MajorAxisLength r.MinorAxisLength r.Eccentricity sind(theta) cosd(theta)];
shapeFeatNames{end+1} = 'region props';

im = 1;


z = ones(size(mask));
img = imrotate(img,-theta(im),'bilinear','loose');
mask = imrotate(mask,-theta(im),'bilinear','loose');
z = imrotate(z,-theta(im),'bilinear','loose');
% check if need to flip upside-down
anys = any(mask,2);
topSum = sum(anys(floor(1:end/2)));
bottomSum = sum(anys)-topSum;
%     bottomSum = sum(sum(mask(:)))-topSum;
if (topSum > bottomSum)
    img = imrotate(img,180,'bilinear','crop');
    mask = imrotate(mask,180,'bilinear','crop');
    z = imrotate(z,180,'bilinear','crop');
end

% after normalizing , find stroke width along x axis (i.e, left-to-right
% strokes)

strokeWidths = sum(imresize(mask,[50 NaN],'nearest'),2);
shapeFeats{end+1} = strokeWidths;
shapeFeatNames{end+1} = 'stroke widths';

% show features with HOG:
% make a bounding box of constant size around this image region.
bb = region2Box(mask);
%bb = inflatebbox(bb,[40 60],'both',true);
bb(4) = bb(2)+32;
center_x = round((bb(1)+bb(3))/2);
width = 32;
bb(1) = center_x-width/2;
bb(3) = center_x+width/2;
bb = round(bb);

img_for_hog = cropper(img,bb);
z_for_hog = cropper(z,bb);
mask_for_hog = cropper(mask & z,bb);


H = fhog(im2single(img_for_hog),4);
mask_for_hog_1 = imResample(double(mask_for_hog),size2(H),'bilinear');
mask_for_hog_1 = repmat(mask_for_hog_1,[1 1 size(H,3)]);
appearanceFeats{end+1} = H;
appearanceFeatNames{end+1} = 'full hog';
H = H.*mask_for_hog_1;
appearanceFeats{end+1} = H;
appearanceFeatNames{end+1} = 'masked hog';
% featureExtractor.bowConf.bowmodel.numSpatialX = [1 1];
% featureExtractor.bowConf.bowmodel.numSpatialY = [1 3];
appearanceFeats{end+1} = featureExtractor.extractFeatures(imresize(img,2,'bilinear'),imresize(mask,2,'nearest'),'normalization','Improved');
appearanceFeatNames{end+1} = 'rotated obj. fisher appearance';


feats.shapeFeats = shapeFeats;
feats.shapeFeatNames = shapeFeatNames;
feats.appearanceFeats = appearanceFeats;
feats.appearanceFeatNames = appearanceFeatNames;

if (debug_)
    clf;
    V = hogDraw( H, 15, 1);
    mm = 2;
    nn = 3;
    subplot(mm,nn,1);
    %imagesc2(occupancyMask);plotPolygons(mm_,'g+');
    
    imagesc2(img_orig);plot(30,30,'g+','MarkerSize',10,'LineWidth',2);
    
    hold on; plotBoxes(mouthBox,'y-');
    plotBoxes(boxes_n,'r-');
    plotBoxes(bbb);
    subplot(mm,nn,2);
    imagesc2(occupancyMask);
    %     hold on; plotBoxes(bb);
    subplot(mm,nn,3); displayRegions(img_for_hog,mask_for_hog);
    subplot(mm,nn,4); imagesc2(V);
    subplot(mm,nn,5); imagesc2(img_orig); title('orig');
    subplot(mm,nn,6); imagesc2(boxSums);
    %     subplot(mm,nn,6); imagesc2(img);
    % find the angle which captures most well
end



end