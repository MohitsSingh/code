function curFeats = getPatchAttributes(im)
curFeats = [];
debug_ = false;
ZZZ= {};

if (debug_)
    close all;
end


% do multiple segmentations

c = {};
r = {};

% segParam =15;
% lengths = zeros(size(segParam));
k = 0;

im = im2uint8(im);
labels = getMultipleSegmentations(im);
for ipp = 1:length(labels)
    k = k+1;    
    subplot(1,3,1);imagesc(im);
    subplot(1,3,2);
    imagesc(labels{k});
    E = edge(rgb2gray(im2double(im)),'canny');
    subplot(1,3,3);
    imagesc(E);            
    pause;
    [~,c_] = paintSeg(im,labels{k});
    c{k} = [[c_{1}.MeanIntensity] ; [c_{2}.MeanIntensity] ; [c_{3}.MeanIntensity]]'/255;    
    r{k} =  regionprops(labels{k},'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
    'MinorAxisLength','Image','Solidity');
end
% for ipp = 1:length(segParam)
%     k = k+1;
%     [~, labels, ~, ~, ~] = vl_quickseg(im, .8,6,segParam(ipp));    
%     imagesc(labels);
%     pause;
%     [~,c_] = paintSeg(im,labels);
%     c{k} = [[c_{1}.MeanIntensity] ; [c_{2}.MeanIntensity] ; [c_{3}.MeanIntensity]]'/255;    
%     r{k} =  regionprops(labels,'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
%     'MinorAxisLength','Image');    
%     labels = fixLabels(labels);
%     imagesc(labels);
%     pause;
%     k = k+1;    
%     [~,c_] = paintSeg(im,labels);
%     c{k} = [[c_{1}.MeanIntensity] ; [c_{2}.MeanIntensity] ; [c_{3}.MeanIntensity]]'/255;    
%     r{k} =  regionprops(labels,'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
%     'MinorAxisLength','Image');
%     
% end

rprops = cat(1,r{:});
c = cat(1,c{:});
% 
% [~, labels, ~, ~, ~] = vl_quickseg(im, .5,2,10);
% [~, labels1, ~, ~, ~] = vl_quickseg(im, .5,2,15);
% 
% [~,c] = paintSeg(im,labels);
% c = [[c{1}.MeanIntensity] ; [c{2}.MeanIntensity] ; [c{3}.MeanIntensity]]'/255;
% [~,c] = paintSeg(im,labels);
% c = [[c{1}.MeanIntensity] ; [c{2}.MeanIntensity] ; [c{3}.MeanIntensity]]'/255;
% [~,c1] = paintSeg(im,labels1);
% c1 = [[c1{1}.MeanIntensity] ; [c1{2}.MeanIntensity] ; [c1{3}.MeanIntensity]]'/255;
% 
% r =  regionprops(labels,'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
%     'MinorAxisLength');
% r1 = regionprops(labels1,'PixelList','Area','PixelIdxList','BoundingBox','Eccentricity','Orientation','MajorAxisLength',...
%     'MinorAxisLength');
% 
% rprops = [r;r1];




if (debug_)
    figure(1),imagesc(segImage);
    hold on;
    plotBoxes2(startRect([2 1 4 3]));
end

topPoints = zeros(length(rprops),2);
bottomPoints = zeros(length(rprops),2);
for n = 1:length(rprops)
    xy = rprops(n).PixelList;
    [miny iminy] = min(xy(:,2));
    [maxy imaxy] = max(xy(:,2));
    topPoints(n,:) = xy(iminy,:);
    bottomPoints(n,:) = xy(imaxy,:);
end

curFeats = rprops;
for k = 1:length(curFeats)
    curFeats(k).topPoints = topPoints(k,:);
    curFeats(k).bottomPoints = bottomPoints(k,:);
        
%     if (k <= size(c,1))
curFeats(k).color = c(k,:);
%     else
% curFeats(k).color = c1(k-size(c,1),:);
%     end
end

