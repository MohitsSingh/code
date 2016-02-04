function [ z ] = paintStats(VOCopts,imageID,Pw,fr,feat,superPix,removeOutliers)
%PAINTSTATS Summary of this function goes here
%   Detailed explanation goes here
I = readImage(VOCopts,imageID);
z = zeros(size(I,1),size(I,2));
q = sub2ind(size(z),fr(2,:),fr(1,:));

% sum for each superpixel it's values, + a dummy values for
% zero assigned pixels....
superPix = double(superPix);
n = zeros(size(unique(superPix(:))))';
subs = [ones(1,length(q));superPix(q)]';
vals = Pw(feat);
sz = size(n);
sp_val = accumarray(subs,vals,sz);
r = regionprops(superPix,'PixelIdxList','Area');%,'Eccentricity','MinorAxisLength');
% % z0 = z;
% % z1 = z;
% % z2 = z;
for m = 1:length(r)
    z(r(m).PixelIdxList) = sp_val(m)/r(m).Area;
    %     z0(r(m).PixelIdxList) = r(m).Eccentricity;
    %     z1(r(m).PixelIdxList) = r(m).Area;
    %     z2(r(m).PixelIdxList) = r(m).MinorAxisLength;
end

% figure(5);imshow(z0);title('Eccentricity');
% figure(5);imagesc(z1);title('Area');

% experimental - throw away outliers...
if (removeOutliers)
    
    % figure,imagesc(t);
    %     z_ = z;
    %     z(z0(:)>.8 | z1(:) < 200 | z2(:) < 4) = mean(z(:));
    %       s = std(z(:));
    %     m = mean(z(:));
    %     t = z>m+3*s;
    %     z(t) = mean(z(:));
    z(z==max(z(:))) = mean(z(:));
    % figure(1);imagesc(z);title('fixed');figure(2);imagesc(z_);
    
end
% z = imfilter(z,fspecial('gauss',9,7));
% z = histeq(z);
% figure(2);imagesc(adapthisteq(z_,'NumTiles',[2 2],'Distribution','rayleigh'));title('adapt');

end

