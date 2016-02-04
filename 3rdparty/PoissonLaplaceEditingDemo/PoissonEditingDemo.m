%This is a toy example to demonstrate the idea of Poisson editing as explain in
%the paper by Patrick P�erez, Michel Gangnet and Andrew Blake: "Poisson
%image editing", Proceeding SIGGRAPH '03 ACM SIGGRAPH 2003 Papers, Pages 313-318 

close all;
clear;
TargIm      = imread('OL.jpg');
SourceIm    = imread('sun.jpg');
SourceMask  = rgb2gray(imread('R.jpg'));
mask_thresh = graythresh(SourceMask);%Using Otsu's threshold segmentation to weed out Jpeg artifacts
SourceMask  = SourceMask > mask_thresh*max(SourceMask(:));

%% Show Source image where we are cutting from
[SrcBoundry, ~] = bwboundaries(SourceMask, 8);
figure, imshow(SourceIm), axis image
hold on
for k = 1:length(SrcBoundry)
    boundary = SrcBoundry{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end
title('Source image - for cutting from');

%% paste sun in target
shift_in_target_image = [700, 100];%xy
%calc bb of mask
[TargImRows, TargImCols, ~] = size(TargIm);
MaskTarg = calc_mask_in_targ_image(SourceMask, TargImRows, TargImCols, shift_in_target_image);
TargBoundry = bwboundaries(MaskTarg, 8);

%% Show where we are going to paste
figure, imshow(TargIm), axis image
hold on
for k = 1:length(TargBoundry)
    boundary = TargBoundry{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 1)
end
title('Target Image with intended place for pasting Source');

%% paste Laplacian of source into Target image
[MaskTarg, TargImPaste] = paste_source_into_targ(SourceIm, TargIm, SourceMask, shift_in_target_image);
figure, imagesc(uint8(TargImPaste)), axis image, title('Target image with laplacian of source inserted');

%% Solve POisson equations in target image wihtihn masked area
TargFilled = PoissonColorImEditor(TargImPaste, MaskTarg);

%% Show end results
figure;
subplot(1, 2, 1)
imshow(SourceIm), axis image
hold on
for k = 1:length(SrcBoundry)
    boundary = SrcBoundry{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end
title('Source Image with intended place for cutting from');
subplot(2, 2, 2)
imshow(TargIm), axis image
hold on
for k = 1:length(TargBoundry)
    boundary = TargBoundry{k};
    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 1)
end
title('Target Image with intended place for pasting Source');
subplot(2, 2, 4)
imshow(uint8(TargFilled));
axis image
title('With Sun implanted')

