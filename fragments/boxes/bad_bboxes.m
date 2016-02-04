function [ bads] = bad_bboxes( boxes, globalOpts)
%BAD_BBOXES Summary of this function goes here
%   Detailed explanation goes here

[~,~,a] = BoxSize(boxes);
bads = a < globalOpts.minBboxArea;

lengths = boxes(:,4)-boxes(:,2)+1;
heights = boxes(:,3)-boxes(:,1)+1;
bads = bads | lengths <= 25;
bads = bads | heights <= 25;

% aspect_ratios = heights./lengths;
% 
% bads = bads | isinf(aspect_ratios);
% bads = bads | isnan(aspect_ratios);
% % bads = bads | aspect_ratios >=6; % Empiric value.
% % bads = bads | aspect_ratios <=.14; % Empiric value.
% 
% bads = bads | aspect_ratios >=2; % Empiric value.
% bads = bads | aspect_ratios <=.14; % Empiric value.
% % bads = bads | aspect_ratios >=4; % Empiric value.

% bads = bads | aspect_ratios >=3; % Empiric value.
% bads = bads | aspect_ratios <=.3; % Empiric value.

end