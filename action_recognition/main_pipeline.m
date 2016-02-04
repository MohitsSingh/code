function [ res ] = main_pipeline(imageData,conf,testMode)
%MAIN_PIPELINE apply all pipeline stages to image.
%   Detailed explanation goes here
    res = struct('valid',false);
    
    % 1. Detect faces
    % 2. For each face, find facial landmarks
    % 3. Detect candidate regions which overlap, at least slightly, with
    % the mouth region
    

end

