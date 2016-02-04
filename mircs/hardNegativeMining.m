function classifier_data = hardNegativeMining(trainData,labels,lambdas,valids)

% EXERCISE4
% Training cofiguration
%targetClass = 1 ;
%targetClass = 'prohibitory' ;
if nargin < 4
    valids = true(1,length(labels));
end
targetClass = 1;
%targetClass = 'danger' ;
numHardNegativeMiningIterations = 7 ;
nPos = sum(labels==1);
schedule = [1 2 5 5 100 100 100 nPos nPos nPos] ;
numHardNegativeMiningIterations = length(schedule);
% Initial positive and negative data

% sel_neg = labels==-1;
sel_neg = find(labels==-1);
pos = trainData(:,ismember(labels,targetClass)) ;
neg = zeros(size(pos,1),0);
for t=1:numHardNegativeMiningIterations
    numPos = size(pos,4) ;
    numNeg = size(neg,4) ;
    C = 1 ;
    lambda = 1 / (C * (numPos + numNeg)) ;
    
    fprintf('Hard negative mining iteration %d: pos %d, neg %d\n', ...
        t, numPos, numNeg) ;
    
    % Train an SVM model (see Step 2.2)
    x = cat(2, pos, neg) ;
    y = [ones(1, size(pos,2)) -ones(1, size(neg,2))] ;
    [w b] = vl_svmtrain(x,y,lambda,'epsilon',0.01,'verbose') ;
    %w = single(reshape(w, modelHeight, modelWidth, [])) ;
    
    curScores = w'*trainData;

    scores_neg = curScores(sel_neg);
    [r,ir] = sort(scores_neg,'descend');
    ir = ir(1:min(length(ir),schedule(t)));
    cur_sel_neg = sel_neg(ir);
    moreNeg = trainData(:,cur_sel_neg);
    assert(all(labels(cur_sel_neg)==-1));
%     [matches, moreNeg] = ...
%         evaluateModel(...
%         vl_colsubset(trainImages', schedule(t), 'beginning'), ...
%         trainBoxes, trainBoxImages, ...
%         w, hogCellSize, scales) ;
    
    % Add negatives
    neg = cat(2, neg, moreNeg) ;
    
    % Remove negative duplicates
    %z = reshape(neg, [], size(neg,4)) ;
    [~,keep] = unique(neg','stable','rows') ;
    neg = neg(:,keep) ;
end


classifier_data.w = [w;b];