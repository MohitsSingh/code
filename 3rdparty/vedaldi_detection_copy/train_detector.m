function w = train_detector(posImages,negImages,param)
% EXERCISE5
% setup ;

% Training cofiguration
targetClass = 1 ;
numHardNegativeMiningIterations = 5 ;
schedule = [1 2 10 40 80] ;

% Scale space configuration
hogCellSize = param.cellSize;
% minScale = -1 ;
minScale = 1;
maxScale = 1;
%maxScale = 3 ;
numOctaveSubdivisions = 3 ;
if (minScale == maxScale)
    scales = minScale;
else
    
    scales = 2.^linspace(...
        minScale,...
        maxScale,...
        numOctaveSubdivisions*(maxScale-minScale+1)) ;
end

trainBoxPatches = {} ;
trainBoxImages = {} ;
trainBoxLabels = [] ;

% Construct negative data
% negImages = getNonPersonImageList();
% Construct positive data
%
for i=1:numel(posImages)
    trainBoxes(:,i) = [.5 .5 fliplr(size2(posImages{i}))]';
    trainBoxPatches{i} = posImages{i};
    trainBoxImages{i} = posImages{i} ;
    trainBoxLabels(i) = 1 ;
end
trainBoxPatches = cat(4, trainBoxPatches{:}) ;

% Compute HOG features of examples (see Step 1.2)
trainBoxHog = {} ;
for i = 1:size(trainBoxPatches,4)
    trainBoxHog{i} = vl_hog(trainBoxPatches(:,:,:,i), hogCellSize) ;
%     trainBoxHog{i} = fhog(trainBoxPatches(:,:,:,i), hogCellSize) ;
end
trainBoxHog = cat(4, trainBoxHog{:}) ;
modelWidth = size(trainBoxHog,2) ;
modelHeight = size(trainBoxHog,1) ;

% -------------------------------------------------------------------------
% Step 5.2: Visualize the training images
% -------------------------------------------------------------------------


if (param.debugging)
    figure(1) ; clf ;
    
    subplot(1,2,1) ;
    imagesc(vl_imarraysc(trainBoxPatches)) ;
    axis off ;
    title('Training images (positive samples)') ;
    axis equal ;
    
    subplot(1,2,2) ;
    imagesc(mean(trainBoxPatches,4)) ;
    box off ;
    title('Average') ;
    axis equal ;
end
% -------------------------------------------------------------------------
% Step 5.3: Train with hard negative mining
% -------------------------------------------------------------------------

% Initial positive and negative data
pos = trainBoxHog(:,:,:,ismember(trainBoxLabels,targetClass)) ;
neg = zeros(size(pos,1),size(pos,2),size(pos,3),0) ;

for t=1:numHardNegativeMiningIterations
    numPos = size(pos,4) ;
    numNeg = size(neg,4) ;
    C = 1 ;
    lambda = 1 / (C * (numPos + numNeg)) ;
    
    fprintf('Hard negative mining iteration %d: pos %d, neg %d\n', ...
        t, numPos, numNeg) ;
    
    % Train an SVM model (see Step 2.2)
    x = cat(4, pos, neg) ;
    x = reshape(x, [], numPos + numNeg) ;
    y = [ones(1, size(pos,4)) -ones(1, size(neg,4))] ;
    w = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;
    w = single(reshape(w, modelHeight, modelWidth, [])) ;
    
    if (param.debugging)
        
        % Plot model
        figure(2) ; clf ;
        %imagesc(hogDraw(w,15,true));
        imagesc(vl_hog('render',w));
        colormap gray ;
        axis equal ;
        title('SVM HOG model') ;
        drawnow;
        figure(3) ;
    end
    % Evaluate on training data and mine hard negatives
    
    [matches, moreNeg] = ...
        evaluateModel(...
        vl_colsubset(negImages, schedule(t), 'random'), ...
        trainBoxes, trainBoxImages, ...
        w, hogCellSize, scales, param.debugging) ;
    
    % Add negatives
    neg = cat(4, neg, moreNeg) ;
    
    % Remove negative duplicates
    z = reshape(neg, [], size(neg,4)) ;
    [~,keep] = unique(z','stable','rows') ;
    neg = neg(:,:,:,keep);
end
