function [features info] = DenseSift(im, nP, sigma, nS, offset)
% [features info] = Image2DenseSift(im, nP, sigma, offset)
%
% Function for extracting Dense Sift features. The features
% can be made spatial using the info structure by Feature2Spatial
%
% im:           Gray-scale image
% nP:           Size of subregions for Sift
% sigma:        Scale of Gaussian Derivative
% nS:           Number of subregions for Sift: nS = 4 yields a 4 x 4 SIFT
% offset:       Top-left border which is removed from image (optional,
%               allows for denser sampling.)
%
% features:     N x 8 matrix with (subregion) SIFT features.
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

[features info] = DenseSiftP(im, nP, sigma, nS, offset);