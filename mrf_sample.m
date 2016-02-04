% graph-cup sample:

% use vl-feat for superpixels
vl_path = '/home/amirro/code/3rdparty/vlfeat-0.9.16/toolbox';
addpath(vl_path);
vl_setup

% graph-cut utilities and others
addpath('/home/amirro/code/utils/graph_cuts');
addpath('/home/amirro/code/utils/');

% graph-cut:
addpath('/home/amirro/code/3rdparty/GCMex');

%% read image 
I = im2single(imread('2007_000170.jpg'));

%% segment into superpixels (use LAB color space)
I_ = single(vl_xyz2lab(vl_rgb2xyz(I)));
segs = vl_slic(I,50,.1);
segs = RemapLabels(segs); % fix labels to range from 1 to n, otherwise a mex within constructGraph crashes. 
segImage = paintSeg(I,segs);

%% make a sample probability map (unary factor for the graph);
[X,Y] = meshgrid(1:size(I,2),1:size(I,1));
g_center = [237 126];
pMap = exp(-((X-g_center(1)).^2+(Y-g_center(2)).^2)/(5*10^3));
%%
killBorders = 0; %useful : zero out probabilities of superpixels within 1 pixels of image borders. 
graphData = constructGraph(I,pMap,segs,killBorders);
edgeParam = .1;  % A very important number!!!! must be > 0 
[labels,labelImage] = applyGraphcut(segs,graphData,edgeParam); 
% display results...
subplot(2,2,1); imagesc(I); axis image;
subplot(2,2,2); imagesc(pMap); axis image; title('unary factor');
subplot(2,2,4); imagesc(graphData.seg_probImage); axis image; title('unary factors (superpix. averaging)');
subplot(2,2,3); imagesc(labelImage); axis image; title('mrf result');





