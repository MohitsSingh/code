function [res,res_bd,resizeRatio,sp_data] = extractSaliencyMap(srcImg,opts)

if (ischar(srcImg))
    srcImg = imread(srcImg);
end
maxImageSize = opts.maxImageSize;
if ~isfield(opts,'useSP')
    useSP = true;           %You can set useSP = false to use regular grid for speed consideration
else
    useSP = opts.useSP;
end
%% 1. Parameter Settings
%doFrameRemoving = true;
doFrameRemoving = false;


% make sure area
imgArea = size(srcImg,1)*size(srcImg,2);
minArea = maxImageSize^2;

%maxSize = max(size(srcImg,1),size(srcImg,2));
resizeRatio = (minArea/imgArea)^.5;
resizeRatio = min(1,resizeRatio);
%resizeRatio = maxImageSize/maxSize;

% resizeRatio=1;

srcImg = imresize(srcImg,resizeRatio,'bilinear');
if doFrameRemoving
    [noFrameImg, frameRecord] = removeframe(srcImg, 'sobel');
    [h, w, chn] = size(noFrameImg);
else
    noFrameImg = srcImg;
    [h, w, chn] = size(noFrameImg);
    frameRecord = [h, w, 1, h, 1, w];
end

%% Segment input rgb image into patches (SP/Grid)
pixNumInSP = opts.pixNumInSP; % was 600                           %pixels in each superpixel
spnumber = round( h * w / pixNumInSP );     %super-pixel number for current image

if useSP
    [idxImg, adjcMatrix, pixelList] = SLIC_Split(noFrameImg, spnumber);
else
    [idxImg, adjcMatrix, pixelList] = Grid_Split(noFrameImg, spnumber);
end
sp_data = struct;
sp_data.idxImg = idxImg;
sp_data.adjcMatrix = adjcMatrix;
sp_data.pixelList = pixelList;

%% Get super-pixel properties
spNum = size(adjcMatrix, 1);
meanRgbCol = GetMeanColor(noFrameImg, pixelList);
meanLabCol = colorspace('Lab<-', double(meanRgbCol)/255);
meanPos = GetNormedMeanPos(pixelList, h, w);
bdIds = GetBndPatchIds(idxImg);
colDistM = GetDistanceMatrix(meanLabCol);
posDistM = GetDistanceMatrix(meanPos);
[clipVal, geoSigma, neiSigma] = EstimateDynamicParas(adjcMatrix, colDistM);

%% Saliency Optimization
[bgProb, bdCon, bgWeight] = EstimateBgProb(colDistM, adjcMatrix, bdIds, clipVal, geoSigma);
wCtr = CalWeightedContrast(colDistM, posDistM, bgProb);
optwCtr = SaliencyOptimization(adjcMatrix, bdIds, colDistM, neiSigma, bgWeight, wCtr);
res = makeSaliencyMap(optwCtr, pixelList, frameRecord, false);
res_bd = makeSaliencyMap(bdCon * 30 / 255, pixelList, frameRecord, true);
if (opts.show)
    clf;subplot(1,3,1);imagesc(srcImg); axis image;title('img');
    subplot(1,3,2); imagesc(res); axis image; colormap(gray);title('saliency');
    subplot(1,3,3); imagesc(1-res_bd); axis image; colormap(gray);title('bd conn.');
end

end

