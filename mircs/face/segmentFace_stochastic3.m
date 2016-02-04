function [Z0,w,b] = segmentFace_stochastic2(im,nIters,roi)
[X,Y] = meshgrid(1:size(im,1),1:size(im,2));
imcenter =size(im)/2;
dist_center = (X-imcenter(1)).^2+(Y-imcenter(2)).^2;
%%

debug_ = true;

if (nargin < 3)
    r1 = 20;
    r2 =35;
    im_pos = dist_center.^.5<=r1;
    im_neg = dist_center.^.5>r2;
else
    im_pos = roi;
    im_neg = ~roi;
end

% figure,imagesc(im_pos)

% start with a generative model!
imcolors = reshape(im2double((im)),[],size(im,3));
colors_pos = imcolors(im_pos(:),:);

obj = gmdistribution.fit(colors_pos,1);



h = pdf(obj,imcolors);
Z0 = reshape(h,size(im,1),size(im,2));
% figure,imagesc(Z0)

% [Z0,w,b] = getPosteriorMap(im,im_pos,[]);
if (nargin < 2)
    nIters = 0;
end
if (nIters == 0)
    return;
end


%
if (debug_)
    subplot(2,2,1);
    imagesc(Z0);daspect([1 1 1]);colorbar
    subplot(2,2,2);
    imshow(im);
    
    disp('pause');
    pause;
end
for t = 1:nIters
    Z0= Z0/max(Z0(:));
    
    n_samples = 150;
    si_pos = weightedSample(1:numel(Z0), Z0(:).^2, n_samples);
    si_neg = weightedSample(1:numel(Z0), 1-Z0(:).^2, n_samples);
    im_pos = false(size(Z0));
    im_neg = false(size(Z0));
    im_pos(si_pos) = 1;
    im_neg(si_neg) = 1;
    im_neg = im_neg & ~ im_pos;
    
    [Z0,w,b] = getPosteriorMap(im,im_pos,im_neg);
    if (debug_)
        
        subplot(2,2,3);
        im_pos_vis = cat(3,im_pos,im_pos,im_pos);
        imshow(im_pos_vis.*im2double(im));
        % axis equal; colorbar
        subplot(2,2,4);
        im_neg_vis = cat(3,im_neg,im_neg,im_neg);
        imshow(im_neg_vis.*im2double(im));
        
        
        subplot(2,2,1);
        imagesc(Z0);daspect([1 1 1]);colorbar
        % axis equal; colorbar
        subplot(2,2,2);
        imshow(im);          
        
        disp('pause');
        pause;
        
    end
end
