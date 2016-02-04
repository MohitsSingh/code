function [res,feats] = ...
    train_kp_detector2(curImgs,sel_train,sel_val,all_kps,kpNames,featureExtractor,feats)

if nargin < 7 
    feats = featureExtractor.extractFeaturesMulti(curImgs,false);
    feats = normalize_vec(feats);
end

addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
%%
% feats_orig = feats_l2;
% feats_l2 = normalize_vec(feats_l2,1);
%%
lambdas = fliplr(logspace(-6,-1,5));
lambdas = [0.0056    0.0003];
% lambdas = fliplr(.00001);
% lambdas = 1;
res = struct('lambda',{},'predictors',{},'errors',{},'error_mean',{});
for iLambda = 1:length(lambdas)
    iLambda
    curLambda = lambdas(iLambda);
    curPredictors = train_predictors(feats,sel_train,all_kps,kpNames,curLambda);
    cur_set = sel_val;
    preds_xy = apply_predictors(curPredictors,feats,cur_set);
    res(iLambda).predictors = curPredictors;
    res(iLambda).lambda = curLambda;
    predictionErrors = measurePredictionError(preds_xy,all_kps(cur_set,:,:));
    res(iLambda).errors = predictionErrors;
    res(iLambda).error_mean = cellfun(@(x) mean(x(~isnan(x))),predictionErrors);
end
