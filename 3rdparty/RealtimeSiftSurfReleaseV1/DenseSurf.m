function [features info] = DenseSurf(im, nP, sR, nS, offset)
% [features info] = DenseSurf(im, nP, sR, nS, offset)
%
% Function for extracting Dense Surf features.
%
% im:           Gray-scale image
% nP:           Number of haar-features to sum.
% sR:           sample rate. Haar features are 2*sR by 2*sR in size
% nS:           Number of subregions for SURF
% offset:       Top-left border which is removed from image (optional,
%               allows for denser sampling.)
%
% features:     N x (4 * nS) haar features
% info:         info structure containing:
%       n:      number of haar features in row direction
%       m:      number of features in col direction
%       row:    row coordinate per haar feature
%       col:    col coordinate per haar feature
%
% Copyright (c) Jasper Uijlings / University of Amsterdam

if nargin < 5
    offset = [0 0];
end

[features info] = DenseSurfP(im, nP, sR, nS, offset);