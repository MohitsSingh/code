function [ output_args ] = train_bow( input_args )
%TRAIN_BOW Summary of this function goes here
%   Detailed explanation goes here

model.classes = {'1','2'};
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
model.classify = @classify ;


histPath = 'hist.mat';
if ~exist(histPath,'file')
    hists = {} ;
    for ii = 1:length(images)
        % for ii = 1:length(images)
        fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
        im = imread(images{ii});
        hists{ii} = getImageDescriptor(model, im);
    end
    
    hists = cat(2, hists{:}) ;
    save(histPath, 'hists') ;
else
    load(histPath) ;
end

end

