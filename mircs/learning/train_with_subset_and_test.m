function [ test_results ] = train_with_subset_and_test(iClass,train_features,feature_subset,...
    train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas,toBalance)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (~exist('lambdas','var'))
    lambdas = logspace(-4,-1,3);
end
if (~exist('toBalance','var'))
    toBalance = 0;
end
test_results = struct;
%debug_range = 1:20:4096;
train_features_1 = transform_features(train_features(feature_subset),train_params.features);
debug_range = 1:1:size(train_features_1);
valids_train = valids(1:length(train_ids));
res_train = train_classifiers(train_features_1(debug_range,valids_train),train_labels(valids_train),train_params,toBalance,lambdas);
test_features_1 = transform_features(test_features(feature_subset),train_params.features);
res_test = apply_classifiers(res_train,test_features_1(debug_range,:),test_labels,train_params);
test_results.target_class = iClass;
test_results.feature_subset = feature_subset;
test_results.performance = res_test;
test_results.classifier_data = res_train.classifier_data;

% test_results = struct('target_class',{},'feature_subset',{},'performance',{},'classifier_data',{});
end