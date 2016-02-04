function showPredsHelper(imdb,L,k,n)
if nargin < 4
    n = 1;
end
curPreds = L.preds{k};
curScores = L.scores{k};
I = imdb.images_data{k};
I = single(imResample(I,[384 384],'bilinear'));
curScoresSoftMax = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
showPredictions(I,curPreds,curScoresSoftMax,L.labels,n);
