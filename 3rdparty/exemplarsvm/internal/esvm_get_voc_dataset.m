function dataset_params = esvm_get_voc_dataset(VOCYEAR,datadir,result_dir)
% Get the dataset structure for a VOC dataset, given the VOCYEAR
% string which is something like: VOC2007, VOC2010, etc.  This assumes
% that VOC is locally installed, see Exemplar-SVM project page for
% instructions if you need to do this.
%
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if ~exist('VOCYEAR','var')
  fprintf(1,'WARNING: using default VOC2007 dataset\n');
  VOCYEAR = 'VOC2007';
end

if ~exist('result_dir','var')
  result_dir = '';
end
% Create a root directory
dataset_params.devkitroot = [result_dir '/'];

% change this path to a writable local directory for the example code
dataset_params.localdir = [dataset_params.devkitroot];

if length(result_dir) == 0
  dataset_params.localdir = '';
end

% change this path to a writable directory for your results
dataset_params.resdir = [dataset_params.devkitroot ['/' ...
                    'results/']];

%This is location of the installed VOC datasets
dataset_params.datadir = datadir;

%Some VOC-specific dataset stats
dataset_params.dataset = VOCYEAR;
dataset_params.testset = 'test';

%NOTE: this is if we want the resulting calibrated models to have
%different special characters in their name
%dataset_params.subname = '';

%Do not skip evaluation, unless it is VOC2010
dataset_params.SKIP_EVAL = 0;

%If enabled, shows outputs throughout the training/testing procedure
dataset_params.display = 0;

%Fill in the params structure with VOC-specific stuff
dataset_params = VOCinit(dataset_params);
