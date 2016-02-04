function [hists,rdata,regions] = feat_extraction(filename)
% function [hists,rdata,regions] = feat_extraction(filename)
%
% This function inputs the filename of an image, and outputs a bag of
% gPb-based region features.
% 
% Related functions: combine_regions, compute_hists
%
% Copyright @ Chunhui Gu, April 2009

regions  = combine_regions(filename);
[hists,rdata] = compute_hists(filename,regions);