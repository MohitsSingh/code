function [curCenter,curBox,curSubWindow,curX] = ...
    predict_next(prevCenter,curIm,patchToFaceRatio,subWindowSize,local_model,...
    alpha_)
sz = size(curIm,1);
curBox = round(inflatebbox([prevCenter prevCenter],sz*patchToFaceRatio,'both',true));
curSubWindow = cropper(curIm,curBox);
curX = getImageStackHOG(curSubWindow,subWindowSize,true,false,8);
curPredictedDeviation = sz*apply_kp_regressor(curX,local_model)';
%                 curPredictedDeviation = curPredictedDeviation/norm(curPredictedDeviation);
curCenter = prevCenter - alpha_*curPredictedDeviation;
% curPredictedDeviation = X_pred_local(:,u)*size(curIm,1)';