function [ test_results ] = train_with_subset_and_test_3(train_features,...
    train_params,train_labels,test_labels,test_features,lambdas,toBalance)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (~exist('lambdas','var'))
    lambdas = logspace(-4,-1,3);
end
if (~exist('toBalance','var'))
    toBalance = 0;
end
test_results = struct;
res_train = train_classifiers(train_features,train_labels,train_params,toBalance,lambdas);
res_test = apply_classifiers(res_train,test_features,test_labels,train_params);
test_results.performance = res_test;
test_results.classifier_data = res_train.classifier_data;
end