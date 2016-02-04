function labelImage = local_segmentation(I,bbox,segs,ucmMAP)
I = im2single(I);

%% segment into superpixels (use LAB color space)
if (nargin < 3)
    I_ = single(vl_xyz2lab(vl_rgb2xyz(I)));
    segs = vl_slic(I_,.1*size(I,1),1);
    segs = RemapLabels(segs); % fix labels to range from 1 to n, otherwise a mex within constructGraph crashes.
end
segImage = paintSeg(I,segs);

%% make a sample probability map (unary factor for the graph);

% maybe the maks is already given
if (numel(bbox) == 4)
    [X,Y] = meshgrid(1:size(I,2),1:size(I,1));
    g_center = (bbox(1:2)+bbox(3:4))/2;
    g_sigma = mean((bbox(3:4)-bbox(1:2)))/.1;
    pMap = exp(-((X-g_center(1)).^2+(Y-g_center(2)).^2)/g_sigma);
    pMap = pMap/max(pMap(:));
else
    %     bbox = imerode(bbox,ones(7));
    pMap = bbox;
    % % % %     pMap = exp(-bwdist(bbox).^2/150);
end
% pMap(pMap<.09) = 0;


%%
killBorders = 1; %useful : zero out probabilities of superpixels within 1 pixels of image borders.
graphData = constructGraph(I,pMap,segs,killBorders);
edgeParam = .5;  % A very important number!!!! must be > 0
[labels,labelImage] = applyGraphcut(segs,graphData,edgeParam);
% display results...
clf;
subplot(2,2,1); imagesc(I); axis image; hold on; plotBoxes2(bbox(:,[2 1 4 3]),'g');
subplot(2,2,2); imagesc(pMap); axis image; title('unary factor');hold on; plotBoxes2(bbox(:,[2 1 4 3]),'g');
subplot(2,2,4); imagesc(graphData.seg_probImage); axis image; title('unary factors (superpix. averaging)');
subplot(2,2,3); imagesc(labelImage); axis image; title('mrf result');
displayRegions(I,{labelImage},0,-1)



