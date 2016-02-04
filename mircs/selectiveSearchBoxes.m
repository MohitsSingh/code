function [ all_boxes,all_blobs ] = selectiveSearchBoxes( im )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
colorTypes = {'Hsv', 'Lab'};
all_boxes = {};
all_blobs = {};
for iColorType = 1:length(colorTypes)
    colorType = colorTypes{iColorType}; % Single color space for demo
    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
    
    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    %k = 50; % controls size of segments of initial segmentation.
    k = 100; % controls size of segments of initial segmentation.
    minSize = k;
    %sigma = 0.8;
    sigma = .5;
    
    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
    
    all_boxes{iColorType} = boxes(:,[2 1 4 3]);
    hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1});
    hBlobs = [hBlobs;RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{2})];
    all_blobs{iColorType} = hBlobs;
    
end

end

