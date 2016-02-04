pos = trainBoxFeatures(:,:,:,ismember(trainBoxLabels,targetClass)) ;
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
    
    % Plot model
    figure(2) ; %
    mm=2;nn=4;nLabels=8;
    for tt = 1:nLabels
        vl_tightsubplot(mm,nn,tt);
        imagesc2(w(:,:,tt));colormap('jet'); 
    end
    title('SVM HOG model') ;
   
    % Evaluate on training data and mine hard negatives
    figure(3) ;
        
    [matches, moreNeg] = evaluateModel(vl_colsubset(trainImages, schedule(t),...
        'random'), trainBoxes, trainBoxImages, w, cellSize, scales,LL);
     dpc
    % Add negatives
    neg = cat(4, neg, moreNeg) ;
    
    % Remove negative duplicates
    z = reshape(neg, [], size(neg,4)) ;
    [~,keep] = unique(z','stable','rows') ;
    neg = neg(:,:,:,keep) ;
end
