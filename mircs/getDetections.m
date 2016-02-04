function [dets] = getDetections(conf,val_set,clusters,iter,suffix,...
    toSave,override,rotations)

% runs N detectors on the images in val_set.
%
% The detectors are defined by detectors(k).ws and detectors(k).b,
% where k is the detector index. iter is the iteration number used for caching.
% params are the svm detection parameters.

% Return arguments
% ----------------
% dets - a cell array with each element (bbs) a Mx12 matrix, where the 12 columns correspond
% to the elements returned by esvm_detect_imageset. Each cell corrseponds
% to the image of the same index in val_set
%
% if (~exist('perImageMasks','var'))
%     perImageMasks = [];
% end


if (~exist('suffix','var'))
    if (isfield(conf,'suffix'))
        suffix = conf.suffix;
    else
        suffix = '';
    end
end

if (~exist('toSave','var'))
    toSave = true;
end

if (~exist('override','var'))
    override= false;
end

if (~exist('rotations','var'))
    rotations = 0;
end

detectionsPath = fullfile(conf.cachedir,['detections_' num2str(iter) suffix '.mat']);
if (~override && toSave && exist(detectionsPath,'file'))
    load(detectionsPath);
else
    
    valids = find([clusters.isvalid]);
    models = [];
    for k = 1:length(valids)
        cluster_id = valids(k);
        model = [];
        winsize = conf.features.winsize;
        model.w = reshape(clusters(cluster_id).w,winsize(1),winsize(2),[]);
        model.b =  clusters(cluster_id).b;
        model.hg_size = size(model.w);
        model.init_params = conf.detection.params.init_params;
        models{1}.models_name = 'clustering'; %#ok<*AGROW>
        models{k}.model = model;
    end
    
    dets = {};
    % run all detectors on all images.
    
    %     parfor iImage = 1:length(val_set)
    for iImage = 1:length(val_set)
        currentID = val_set{iImage};
        disp(100*iImage/length(val_set));
        if (ischar(currentID))
            I = getImage(conf,currentID);
        else
            %             imagePath = getImagePath(conf,currentID);
            
            %         else
            %             I = currentID;
            %         end
            I = toImage(conf,currentID);
        end
        
        I_orig = I;
        resstruct= [];
        for iRotation = 1:length(rotations)
            I = imrotate(I_orig,rotations(iRotation),'bilinear','crop');
            [resstruct_,~] = esvm_detect(I, models,conf.detection.params);
            for kk = 1:length(resstruct_.bbs)
                resstruct_.bbs{kk} = [resstruct_.bbs{kk} rotations(iRotation)*ones(size(resstruct_.bbs{kk},1),1)];
            end
            if (isempty(resstruct))
                resstruct = resstruct_;
            else
                for kk = 1:length(resstruct.bbs)
                    resstruct.bbs{kk} =  [resstruct.bbs{kk};resstruct_.bbs{kk}];
                    resstruct.xs{kk} =  [resstruct.xs{k},resstruct_.xs{kk}];
                end
            end
        end
        %         d = conf.detection.params.init_params.sbin; % apparently, this helps...
        %         d = floor(d/2);
        %         [resstruct2,~] =esvm_detect(I(d:end,d:end,:), models,conf.detection.params);
        %         for k = 1:length(resstruct.bbs)
        %             resstruct.bbs{k} =  [resstruct.bbs{k};resstruct2.bbs{k}];
        %             resstruct.xs{k} =  [resstruct.xs{k},resstruct2.xs{k}];
        %         end
        dets{iImage} = resstruct;
    end
    if (toSave)
        save(detectionsPath,'dets','-v7.3');
    end
end