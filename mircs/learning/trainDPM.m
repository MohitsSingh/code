function model = trainDPM(conf,r_true,r_false,cls)

% Train and score a model with 2*n components.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG) features'
% testyear allows you to test on a year other than VOCyear (set in globals.m)

% vlfeat
% addpath('~/code/3rdparty/vlfeat-0.9.14/toolbox');
% vl_setup;

note = 'doing stuff';

globals;

currentDir = pwd;
mkdir(conf.DPM_path);
cd (conf.DPM_path);
mkdir(cachedir);
% record a log of the training procedure
diary([cachedir cls '.log']);

% set the note to the training time if none is given
if nargin < 3
    note = datestr(datevec(now()), 'HH-MM-SS');
end
n =1;
model = model_train(cls, n, note,r_true,r_false);
% lower threshold to get high recall
model.thresh = min(-1.1, model.thresh);
cd(currentDir);