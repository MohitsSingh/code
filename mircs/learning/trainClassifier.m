function [ w b info] = trainClassifier( globalOpts,posFeatureVecs,negFeatureVecs)
%TRAINCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

nPosSamples = size(posFeatureVecs,2);
nNegSamples = size(negFeatureVecs,2);

% toRepeat = round(nNegSamples/nPosSamples);
toRepeat = 10;
posFeatureVecs = repmat(posFeatureVecs,...
    1,toRepeat);
nPosSamples = size(posFeatureVecs,2);


% globalOpts.svm.C = 10;

W = [posFeatureVecs,...
    negFeatureVecs];
clear posFeatureVecs negFeatureVecs;

% remove nans...
A = sum(W);
A = (isnan(A));

W = W(:,~A);
y = [ones(1,nPosSamples),...
    -ones(1,nNegSamples)];

y = y(~A);

% W = full(W);
% psix = W;
% psix = globalOpts.hkmfun(W);
%psix = sparse(hkm(W));

% p = Pegasos(psix', int8(y)','iterNum', 1000,'lambda',1/length(y));
% ll =  10/length(y);
% p = Pegasos(psix', int8(y)','iterNum', 100000,'lambda',[ll ll]);
[w b info] = vl_pegasos(full(W),int8(y)',.0001,...
    'MaxIterations',100,'homkermap',1,'KChi2','Period',.5);
%Pegasos.train(single(full(X(trIdx, :))), y(trIdx), lambda(i));
% vl_pegasos(
% p = Pegasos(W', int8(y)','iterNum', 100);
%iterNum = checkVarargin(varargin, 'iterNum', 1e3);
% 
% model.b = w(end, :) ;
% model.w = w(1:end-1, :);
