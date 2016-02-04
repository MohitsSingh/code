function [ model ] = trainClassifier( globalOpts,trainingInstances)
%TRAINCLASSIFIER Summary of this function goes here
%   Detailed explanation goes here

nPosSamples = size(trainingInstances.posFeatureVecs,2);
nNegSamples = size(trainingInstances.negFeatureVecs,2);


trainingInstances.posFeatureVecs = repmat(trainingInstances.posFeatureVecs,...
    1,round(nNegSamples/nPosSamples));
nPosSamples = size(trainingInstances.posFeatureVecs,2);


% globalOpts.svm.C = 10;

W = [trainingInstances.posFeatureVecs,...
    trainingInstances.negFeatureVecs];

% remove nans...
A = sum(W);
A = (isnan(A));

W = W(:,~A);
y = [ones(1,nPosSamples),...
    -ones(1,nNegSamples)];

y = y(~A);

W = full(W);
psix = globalOpts.hkmfun(W);
%psix = sparse(hkm(W));

% p = Pegasos(psix', int8(y)','iterNum', 1000,'lambda',1/length(y));
% ll =  10/length(y);
% p = Pegasos(psix', int8(y)','iterNum', 100000,'lambda',[ll ll]);

p = Pegasos(psix', int8(y)','iterNum', 100);
%iterNum = checkVarargin(varargin, 'iterNum', 1e3);

w = real(p.w);
model.b = w(end, :) ;
model.w = w(1:end-1, :);
