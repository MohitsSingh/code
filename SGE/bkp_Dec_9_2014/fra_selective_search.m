function res = fra_selective_search(conf,I,reqInfo)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV'));
    return;
end

%%
%S
roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
[rois,roiBox,im,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
roiParams.useCenterSquare = false;

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
all_boxes = {};
all_blobs = {};
for iColorType = 1:length(colorTypes)
    colorType = colorTypes{iColorType}; % Single color space for demo    
    % Here you specify which similarity functions to use in merging
    simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, @SSSimBoxFillOrig, @SSSimSize};
    simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
    
    % Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
    % Note that by default, we set minSize = k, and sigma = 0.8.
    k = 50; % controls size of segments of initial segmentation.
    minSize = k;
    %sigma = 0.8;
    sigma = .5;
        
    % Perform Selective Search
    [boxes blobIndIm blobBoxes hierarchy] = Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
    boxes = BoxRemoveDuplicates(boxes);
        
    all_boxes{iColorType} = boxes(:,[2 1 4 3]);
    %     hBlobs = RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{1});
    %     hBlobs = [hBlobs;RecreateBlobHierarchyIndIm(blobIndIm, blobBoxes, hierarchy{2})];
    %     all_blobs{iColorType} = hBlobs;
    
end
res.boxes = cat(1,all_boxes{:});
% res.blobs = cat(1,all_blobs{:});
