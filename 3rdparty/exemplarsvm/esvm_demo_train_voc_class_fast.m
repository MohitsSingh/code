% DEMO: PASCAL VOC "mini" training/testing script
% This function can generate a nice HTML page by calling:
% publish('esvm_demo_train_voc_class_fast.m','html')
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved. 
%
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm
function [models,M] = esvm_demo_train_voc_class_fast(cls, ...
                                                  data_directory, ...
                                                  dataset_directory, ...
                                                  results_directory)


addpath(genpath(pwd));

if ~exist('cls','var')
  fprintf(1,'esvm_demo_train_fast: defaulting to class=car\n');
  cls = 'car';
end

if ~exist('data_directory','var')
  data_directory = '/Users/tomasz/projects/pascal/';
end

if ~exist('dataset_directory','var')
  dataset_directory = 'VOC2007';
end

if ~exist('results_directory','var')
  %results_directory = '';
  results_directory = sprintf(['/nfs/baikal/tmalisie/esvm-%s-' ...
                    '%s-fast/'], ...
                              dataset_directory, cls);
end

%data_directory = '/Users/tomasz/projects/Pascal_VOC/';
%results_directory = '/nfs/baikal/tmalisie/esvm-data/';

%data_directory = '/csail/vision-videolabelme/people/tomasz/VOCdevkit/';
%results_directory = sprintf('/csail/vision-videolabelme/people/tomasz/esvm-%s/',cls);


dataset_params = esvm_get_voc_dataset(dataset_directory,...
                                      data_directory,...
                                      results_directory);


dataset_params.display = 1;
%dataset_params.dump_images = 1;

% Issue warning if lock files are present
lockfiles = check_for_lock_files(results_directory);
if length(lockfiles) > 0
  fprintf(1,'WARNING: %d lockfiles present in current directory\n', ...
          length(lockfiles));
end

%KILL_LOCKS = 1;
%for i = 1:length(lockfiles)
%  unix(sprintf('rmdir %s',lockfiles{i}));
%end

%% Set exemplar-initialization parameters
params = esvm_get_default_params;
params.model_type = 'exemplar';
params.dataset_params = dataset_params;

%Initialize exemplar stream
stream_params.stream_set_name = 'trainval';
stream_params.stream_max_ex = 1;
stream_params.must_have_seg = 0;
stream_params.must_have_seg_string = '';
stream_params.model_type = 'exemplar'; %must be scene or exemplar;
stream_params.cls = cls;

%Create an exemplar stream (list of exemplars)
e_stream_set = esvm_get_pascal_stream(stream_params, ...
                                      dataset_params);

neg_set = esvm_get_pascal_set(dataset_params, ['train-' cls]);

%Choose a models name to indicate the type of training run we are doing
models_name = ...
    [cls '-' params.init_params.init_type ...
     '.' params.model_type];

initial_models = esvm_initialize_exemplars(e_stream_set, params, models_name);

%% Perform Exemplar-SVM training
train_params = params;
train_params.detect_max_scale = 0.5;
train_params.train_max_mined_images = 50;
train_params.detect_exemplar_nms_os_threshold = 1.0; 
train_params.detect_max_windows_per_exemplar = 100;

%% Train the exemplars and get updated models name
[models,models_name] = esvm_train_exemplars(initial_models, ...
                                            neg_set, train_params);


val_params = params;
val_params.detect_exemplar_nms_os_threshold = 0.5;
val_params.gt_function = @esvm_load_gt_function;

val_set_name = ['trainval+' cls];

val_set = esvm_get_pascal_set(dataset_params, val_set_name);
val_set = val_set(1:40);

%% Apply trained exemplars on validation set
val_grid = esvm_detect_imageset(val_set, models, val_params, val_set_name);

%% Perform Platt calibration and M-matrix estimation
M = esvm_perform_calibration(val_grid, val_set, models, ...
                             val_params);

%% Define test-set
test_params = params;
test_params.detect_exemplar_nms_os_threshold = 0.5;
test_set_name = ['test+' cls];
test_set = esvm_get_pascal_set(dataset_params, test_set_name);
test_set = test_set(1:100);

%% Apply on test set
test_grid = esvm_detect_imageset(test_set, models, test_params, test_set_name);

%% Apply calibration matrix to test-set results
test_struct = esvm_pool_exemplar_dets(test_grid, models, M, test_params);

%% Show top detections
maxk = 20;
allbbs = esvm_show_top_dets(test_struct, test_grid, test_set, models, ...
                       params,  maxk, test_set_name);

%% Perform the exemplar evaluation
[results] = esvm_evaluate_pascal_voc(test_struct, test_grid, params, ...
                                     test_set_name, cls, models_name);
