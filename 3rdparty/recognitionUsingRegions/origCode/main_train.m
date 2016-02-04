function main_train()
% function main_train()
%
% The function loads training/test data split and output training data with
% learned region weights.
%
% Input:    train(test)_name(class) (by loading filename.mat)
% Output:   base_dir:           directory where images are stored
%           train(i).filename:  filename of the i'th training image
%           train(i).categId:   category id of the image
%           train(i).masks:     object support mask(s) in the image
%           train(i).bboxes:    object bounding box(es) in the image
%           train(i).regions:   a set of regions (in binary mask)
%           train(i).weights:   learned weights on the regions
%           train(i).beta:      learned parameter values for logistic fns
%
% Related functions: combine_regions, assign_regionflags, compute_hists,
% dist_bwo, get_featweights, logist2
%
% Copyright @ Chunhui Gu April 2009

% set path
addpath util/
base_dir = 'images/';
run_dir = 'mat/run1/';

%%% Parameter Setting
param.c = 1;            % regularization-margin tradeoff parameter
param.prec = 0.001;     % precision
param.num = 5;          % number of nearest hits/misses per matching
fraction = 0.8;         % fraction of region overlaping with bounding box
%%% End Parameter Setting

% load training/test data split
load([run_dir 'filename.mat']);

tic;
nimgs = length(train_class);
region_flags = cell(1,nimgs);
train_hists = cell(1,nimgs);
train_rdata = cell(1,nimgs);
for id = 1:nimgs,
    
    fname = train_name{id}(length(base_dir)+1:end);
    filename = [base_dir fname];
    categId  = train_class(id);
    bboxes   = load([filename '.ground_truth']);
    masks    = imread([filename '_groundtruth.bmp']) == categId;
    sc_ratio = size(imread(filename),1)/size(imread([filename '_ori.jpg']),1);
    bboxes = round(bboxes * sc_ratio);
    
    regions  = combine_regions(filename);
    region_flags{id} = assign_regionflags(regions,bboxes,fraction);
    [train_hists{id},train_rdata{id}] = compute_hists(filename,regions);
    
    images(id).filename = fname;
    images(id).categId  = categId;
    images(id).bboxes   = bboxes;
    images(id).masks    = masks;
    images(id).regions  = regions;
    
    fprintf('Done exemplar %d/%d in %g seconds.\n',id,nimgs,toc);
end;

% save([run_dir 'hists_pb.mat'],'train_hists','train_rdata');
% save([run_dir 'region_flags.mat'],'region_flags');
% load([run_dir 'hists_pb.mat']);
% load([run_dir 'region_flags.mat']);

tic;
nimgs = length(train_class);
for id = 1:nimgs,
    
    categId  = train_class(id);
    
    % precomputed data
    regionflags = region_flags{id};
    hists = train_hists{id};
    rdata = train_rdata{id};
    
    hists = hists(regionflags,:);
    rdata = rdata(regionflags,:);
    
    distf_map = zeros(size(hists,1),nimgs,'single');
    distf_arg = zeros(size(hists,1),nimgs,'single');
    distb_map = zeros(size(hists,1),nimgs,'single');
    distb_arg = zeros(size(hists,1),nimgs,'single');
    for exempId = 1:nimgs,
        
        % precomputed data
        eregionflags = region_flags{exempId};
        ehists = train_hists{exempId};
        erdata = train_rdata{exempId};

        ehistsf = ehists(eregionflags,:);
        erdataf = erdata(eregionflags,:);
        ehistsb = ehists(~eregionflags,:);
        erdatab = erdata(~eregionflags,:);
                
        map = dist_bwo(hists,ehistsf,rdata,erdataf,'chi-square',true,false);
        [distf,argf] = min(map,[],2);
        map = dist_bwo(hists,ehistsb,rdata,erdatab,'chi-square',true,false);
        [distb,argb] = min(map,[],2);
        
        indf = find(eregionflags);
        indb = find(~eregionflags);
        
        distf_map(:,exempId) = distf;
        distf_arg(:,exempId) = indf(argf);
        distb_map(:,exempId) = distb;
        distb_arg(:,exempId) = indb(argb);
        
        fprintf('Done exemplar %d/%d for id %d in %g seconds.\n',exempId,nimgs,id,toc);
        
    end;
    
    dist_map = [distf_map distb_map];
    labels = [train_class==categId false(1,nimgs)];
    
    w = get_featweights(dist_map,labels,id,'ranklearning',param);
    weights = zeros(size(regionflags));
    weights(regionflags) = w;
    
    try
        dd = w'*dist_map;
        beta = logist2(labels', double([dd',ones(length(labels),1)]));
    catch
        beta = [];
    end;
    
    images(id).weights  = weights;
    images(id).beta     = beta;
    
end;

save([run_dir 'training_data.mat'], 'images', 'base_dir','-v7.3');
