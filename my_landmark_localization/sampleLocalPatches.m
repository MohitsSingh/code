function [kp_patches,deviations,kps_local,kp_patch_centers] = ...
    sampleLocalPatches(imgs,xy,patchToFaceRatio,maxDeviation)

% for u = 1:length(IsTr)
%
%     figure(1);clf;
%     subplot(1,2,1);  imagesc2(IsTr{u});
%     plotPolygons(xy(u,:),'g.');
%     drawnow;
%     pause
% end

sizes = cell2mat(cellfun2(@size2,imgs));% height/width
nImages = length(imgs);
heights = sizes(:,1);
deviations_theta = rand(nImages,1)*2*pi;
deviations_r = heights.*rand(nImages,1)*maxDeviation;
deviations = [deviations_r deviations_r].*[cos(deviations_theta) sin(deviations_theta)];

xy_orig = xy;
xy = xy_orig+deviations;
rects = inflatebbox([xy xy],heights*patchToFaceRatio,'both',true);
kp_patches = multiCrop([],imgs,round(rects));
kps_local = xy_orig-rects(:,[1 2])+1;
kp_patch_centers = xy;
% for u = 1:length(U)
%     m = U{u};
%     figure(1);clf;
%     subplot(1,2,1); imagesc2(m);
%     sz = size2(m);
%     quiver(sz(2)/2,sz(1)/2,-deviations(u,1),-deviations(u,2),0,'g');
%     subplot(1,2,2); imagesc2(IsTr{u});
%     plotPolygons(xy_orig(u,:),'g.');
%     drawnow;
%     pause
% end

