function [ canvas ] = composeRGB( imr, img, imb )

[hr wr] = size(imr);
[hg wg] = size(img);
[hb wb] = size(imb);

if hr ~= hg | hr ~= hb | wr ~= wg | wr ~= wb
    fprintf('Error - mismatched RGB components');
end

canvas = zeros(hr, wr, 3);

canvas(:, :, 1) = imr;
canvas(:, :, 2) = img;
canvas(:, :, 3) = imb;
