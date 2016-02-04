
function seg_mask = getSegments_graphCut_2(im,obj_mask,maxScale,toDisplay,obj_mask_hard)
if (nargin < 5 || isempty(obj_mask_hard))
    obj_mask_hard = false(size(obj_mask));
end
if (nargin < 4)
    toDisplay = false;
end
if (nargin < 3 || isempty(maxScale))
    maxScale = size(im,1);
end
im = imresize(im,[maxScale NaN],'bilinear');

if (islogical(obj_mask))
    obj_mask = imresize(obj_mask,size2(im),'nearest');
    
    R_false = ~imdilate(obj_mask,ones(3));
    %
    %     imshow(R)
    %     [X,Y] = meshgrid(1:size(M,2),1:size(M,1));
    %     g_center = fliplr(size2(M))/2;
    %     g_sigma = mean(size2(M))/.2;
    %     pMap = exp(-((X-g_center(1)).^2+(Y-g_center(2)).^2)/g_sigma);
    %
    %      pMap = R;
    % pMap = double(exp(-bwdist(obj_mask)/5));
    
    % pMap = double(1-exp(-bwdist(~obj_mask)/5));
    
    pMap = im2double(obj_mask);
    % rr = regionprops(obj_mask,'Centroid','MajorAxis','MinorAxis','Orientation');
    % mm = maskEllipse(size(obj_mask,1),size(obj_mask,2),rr.Centroid(2),rr.Centroid(1),rr.MajorAxisLength/2,rr.MinorAxisLength/2,90+pi*rr.Orientation/180);
    % imshow(mm)
    % figure,imshow(obj_mask)
    
    pMap = imfilter(pMap,fspecial('gaussian',[41 41],5));
%     pMap(R_false) = 0;
    pMap = addBorder(pMap,1,0);
%     obj_
%     pMap(obj_mask_hard) = 1;
    pMap = normalise(pMap);
    %     pMap = pMap*.8;
    %     pMap = pMap/max(pMap(:));
    % M1 =  vl_xyz2lab(vl_rgb2xyz(im2single(im)));
    %     seg = st_segment(M1,pMap,.1,5);
else
    obj_mask = imresize(obj_mask,size2(im),'bilinear');
    pMap = obj_mask;
%     pMap = addBorder(pMap,1,0);
end

%[seg_mask,energies] = st_segment(im2uint8(im),pMap,.5,3);
[seg_mask] = st_segment(im2uint8(im),pMap,.5,3);
% energies

segResult = normalise(bsxfun(@times,seg_mask,im));
%     subplot(1,3,1); imagesc(normalise(M)); axis image;
    subplot(1,3,2); imagesc(pMap); axis image;
    subplot(1,3,3);imagesc(segResult); axis image;
%
if (toDisplay)
    displayRegions(segResult,pMap>.5,[],-1);
end

