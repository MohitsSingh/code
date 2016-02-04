function boxes = SelectiveSearchBoxes(im)
% Based on Segmentation as Selective Search for Object Recognition,
% K.E.A. van de Sande, J.R.R. Uijlings, T. Gevers, A.W.M. Smeulders, ICCV 2011

% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
kThresholds = [100 200];
sigma = 0.8;
numHierarchy = length(colorTypes) * length(kThresholds);

% vl_slic

% As an example, use a single Pascal VOC image
% images = {'000015.jpg'};

% VOCinit;
% theSet = 'train';
% [images, labs] = textread(sprintf(VOCopts.imgsetpath, theSet), '%s %s');

% For each image do Selective Search
%fprintf('Performing selective search: ');
tic;
% boxes = cell(1, length(images));
% for iImage=1:length(images)
% if mod(iImage,100) == 0
%     fprintf('%d ', iImage);
% end
idx = 1;
currBox = cell(1, numHierarchy);
% im = imread(sprintf(VOCopts.imgpath, images{iImage})); % For Pascal Data

for k = kThresholds
    minSize = k; % We use minSize = k.
    
    for colorTypeI = 1:length(colorTypes)
        colorType = colorTypes{colorTypeI};
        
        currBox{idx} = SelectiveSearch(im, sigma, k, minSize, colorType);
        idx = idx + 1;
    end
end

% boxes{iImage} = cat(1, currBox{:}); % Concatenate results of all hierarchies
% boxes{iImage} = unique(boxes{iImage}, 'rows'); % Remove duplicate boxes

boxes = cat(1,currBox{:});
boxes = unique(boxes,'rows');

% boxes = unique(boxes,'rows');
% end
% figure;
% for k = 1:length(b)
%     imshow(imcrop(im,b(k,:)));
%     pause(.1);
% end