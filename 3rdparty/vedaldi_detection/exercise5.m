% EXERCISE5
% setup ;
% Training cofiguration
targetClass = 1 ;
numHardNegativeMiningIterations = 5 ;
%schedule = [1 2 10 10 10] ;
% Scale space configuration
cellSize = 8 ;
minScale = -1 ;
maxScale = 3 ;
numOctaveSubdivisions = 3 ;
% numOctaveSubdivisions = 1 ;
scales = 2.^linspace(...
    minScale,...
    maxScale,...
    numOctaveSubdivisions*(maxScale-minScale+1)) ;


LL.imageNames = {fra_db.imageID};
LL.images = imdb.images_data;

% -------------------------------------------------------------------------
% Step 5.1: Construct custom training data
% -------------------------------------------------------------------------
% Load object examples
trainImages = {};
trainBoxLabels = [];
trainBoxes = [];
trainBoxPatches = {} ;
trainBoxImages = {};
for t = 1:length(fra_db)
    if ~fra_db(t).isTrain
        continue
    end
    curLabel = imdb.labels{t};
    p = curLabel>=3;
    if none(p),continue,end
    curBox = region2Box(p);
    curBox =  makeSquare(curBox,true);
    
%     curBox = round(inflatebbox(curBox,3,'both',false));
    V = 5;
    curBox = round(inflatebbox(cat(1,curBox),size(imdb.images_data{t},1)/V,'both',true));
    
%     (size(imdb.images_data{t},1)/V)/64
%     curImg = cropper(imdb.images_data{t},curBox);
    %     curImg = imResample(curImg,[8 8],'bilinear');
    trainBoxes(:,end+1) = curBox;
    trainBoxPatches{end+1} = imResample(cropper(LL.scores_coarse{t},curBox),[8 8],'bilinear');
    %     trainImages{end+1} = curImg,curBox);
    trainBoxLabels(end+1) = 1;
    trainBoxImages{end+1} = fra_db(t).imageID;
    trainImages{end+1} = fra_db(t).imageID;
end


trainBoxPatches = cat(4, trainBoxPatches{:}) ;
trainBoxFeatures = trainBoxPatches;
modelWidth = size(trainBoxFeatures,2) ;
modelHeight = size(trainBoxFeatures,1) ;

% -------------------------------------------------------------------------
% Step 5.2: Visualize the training images
% -------------------------------------------------------------------------

% figure(1) ; clf ;
%
% subplot(1,2,1) ;
% imagesc(vl_imarraysc(trainBoxPatches)) ;
% axis off ;
% title('Training images (positive samples)') ;
% axis equal ;
%
% subplot(1,2,2) ;
% imagesc(mean(trainBoxPatches,4)) ;
% box off ;
% title('Average') ;
% axis equal ;

% -------------------------------------------------------------------------
% Step 5.3: Train with hard negative mining
% -------------------------------------------------------------------------
%
% Initial positive and negative data
%%
schedule = [1 5*ones(1,10)];
numHardNegativeMiningIterations=11;
doTraining;

%%
% -------------------------------------------------------------------------
% Step 5.3: Evaluate the model on the test data
% -------------------------------------------------------------------------

im = imread('data/myTestImage.jpeg') ;
im = im2single(im) ;

% Compute detections
[detections, scores] = detect(im, w, cellSize, scales) ;
keep = boxsuppress(detections, scores, 0.25) ;
detections = detections(:, keep(1:10)) ;
scores = scores(keep(1:10)) ;

% Plot top detection
figure(3) ; clf ;
imagesc(im) ; axis equal ;
hold on ;
vl_plotbox(detections, 'g', 'linewidth', 2, ...
    'label', arrayfun(@(x)sprintf('%.2f',x),scores,'uniformoutput',0)) ;
title('Multiple detections') ;