function [samples,locs] = sampleHogs(conf,ids,suffix,max_samples,save_samples,mask)
%SAMPLEHOGS Summary of this function goes here
%   Detailed explanation goes here

if (nargin < 3)
    if (isfield(conf,'suffix'))
        suffix = conf.suffix;
    else
        suffix = [];
    end
end

if (nargin < 4)
    max_samples = inf;
end

if (nargin < 5)
    save_samples = 1;
end

samplesPath = fullfile(conf.cachedir,['samples_' suffix '.mat' ]);
if (exist(samplesPath,'file') && save_samples)
    load(samplesPath);
    return;
    
end
samples = {};
locs = {};


ws = conf.features.winsize;

for k = 1:length(ids)
    currentID = ids{k};
    if (ischar(currentID))
        disp(currentID);
%         imgPath = getImagePath(conf,currentID);
%         I = toImage(conf,imgPath);
            I = getImage(conf,currentID);
    else
        disp(['sampling from image: ' num2str(k)]);
        I = currentID;
    end
    
    % t.hog are the hogs in different pyramid levels
    % t.scales are the scales of each level...
    t = get_pyramid(I,conf);
    %fsize = esvm_features;
    fsize = 32;
    
    pyr_N = cellfun(@(x)prod([size(x,1) size(x,2)]-ws+1),t.hog);
    sumN = sum(pyr_N);
    X = zeros(prod(ws)*fsize,sumN);
    offsets = cell(length(t.hog), 1);
    scales = cell(length(t.hog), 1);
    uus = cell(length(t.hog),1);
    vvs = cell(length(t.hog),1);
    
    counter = 1;
    for i = 1:length(t.hog)
        s = size(t.hog{i});
        NW = s(1)*s(2);
        ppp = reshape(1:NW,s(1),s(2));
        curf = reshape(t.hog{i},[],fsize);
        b = im2col(ppp,[ws ws]);
        
        offsets{i} = b(1,:);
        offsets{i}(end+1,:) = i;
        
        for j = 1:size(b,2)
            X(:,counter) = reshape (curf(b(:,j),:),[],1);
            counter = counter + 1;
        end
        [uus{i},vvs{i}] = ind2sub(s,offsets{i}(1,:));
    end
    
    % get raw features too, for checking minimum energy...
    
    % sampling method:
    % using raw gradient energy or quantization error.
    
    if (strcmp(conf.clustering.sample_method, 'quant_error'))
        E = getQuantizationError(X,conf.dict).^2;
        if (conf.clustering.windows_per_image >=size(X,2))
            window_inds = 1:size(X,2);
        else
            window_inds = double(weightedSample(X, E,  conf.clustering.windows_per_image));
        end
        offsets = cat(2,offsets{:});
        uus = cat(2,uus{:});
        vvs = cat(2,vvs{:});
        scales = t.scales(offsets(2,:));
        bbs = uv2boxes(conf,uus(),vvs(),scales(),t);
        X = X(:,window_inds);
        %
        
        %%
        
        
    else
        
        conf.detection.params.init_params.features = @features_raw;
        t_raw = get_pyramid(I,conf);
        conf.detection.params.init_params.features = @esvm_features;
        X_raw = zeros(prod(ws)*18,sumN);
        counter = 1;
        for i = 1:length(t_raw.hog)
            s = size(t_raw.hog{i}(:,:,1:18));
            NW = s(1)*s(2);
            ppp = reshape(1:NW,s(1),s(2));
            curf = reshape(t_raw.hog{i}(:,:,1:18),[],18);
            b = im2col(ppp,[ws ws]);
            
            for j = 1:size(b,2)
                X_raw(:,counter) = reshape (curf(b(:,j),:),[],1);
                counter = counter + 1;
            end
        end
        
        offsets = cat(2,offsets{:});
        uus = cat(2,uus{:});
        vvs = cat(2,vvs{:});
        scales = t.scales(offsets(2,:));
        
        has_min_energy = sum(X_raw)/size(X_raw,1) > conf.clustering.min_hog_energy;
        
        X = X(:,has_min_energy);
        
        if (isempty(X))
            continue;
        end
        %     offsets = offsets(:,has_min_energy);
        uus = uus(has_min_energy);
        vvs = vvs(has_min_energy);
        scales = scales(has_min_energy);
        window_inds = vl_colsubset(1:length(uus),...
            conf.clustering.windows_per_image,'Uniform')';
    end
    
    boxes = uv2boxes(conf,uus(window_inds),vvs(window_inds),scales(window_inds),t);
    overlaps = boxesOverlap(boxes);
    
    [ii,jj] = find(overlaps>conf.clustering.max_sample_ovp);
    
    removed = false(size(window_inds));
    for ki = 1:length(jj)
        % only remove a box which overlaps with a box which
        % hasn't been removed yet.
        if (~removed(ii((ki))))
            removed(jj(ki)) = true;
        end
    end
    
    window_inds(removed) = [];
    boxes(removed,:) = []; % just keep the boxes themselves for later visualization
    
% %     norms = E(window_inds);
% %     %%
% %     
% %     R = zeros(size(I,1),size(I,2));
% %     for q = 1:length(window_inds)
% %         %                 k = window_inds(q);
% %         b = round(boxes(q,1:4));
% %         b(1) = max(b(1),1);
% %         b(2) = max(b(2),1);
% %         b(3) = min(b(3),size(I,2));
% %         b(4) = min(b(4),size(I,1));
% %         %R(b(2):b(4),b(1):b(3)) = max(R(b(2):b(4),b(1):b(3)),norms(k));
% %         R(b(2):b(4),b(1):b(3)) = R(b(2):b(4),b(1):b(3)) +1;
% %     end
% %     %
% %     R = R-min(R(:));
% %     R = R/max(R(:));
% %     %     mm = mean(R(:));
% %     %     ss = std(R(:));
% %     %     R = R-mean(R(:)) > .5*ss;
% %     
% %     R = cat(3,R,R,R);
% %     I = im2double(I);
% %     imshow(cat(2,I,R.*I));
% %     pause;
% %     continue;
    
    
    % further subsample a small set of features to accomodate desired
    % number samples
    if (nargin >= 4)
        sel = vl_colsubset(1:size(boxes,1),round(max_samples/length(ids)));
        boxes = boxes(sel,:);
        window_inds = window_inds(sel);
    end
    
    boxes(:,11) = k; % note, this indicates the image from which the
    % box was sampled...
    
    locs{k} = boxes;
    samples{k} = X(:,~removed);
    
end
% samples = cat(2,samples{:});
% locs = cat(1,locs{:});
%
% if (nargin == 4)
%     a = vl_colsubset(1:size(samples,2),max_samples);
%     samples = samples(:,a);
%     locs = locs(a,:);
% end

if (save_samples)
    save(samplesPath,'samples','locs');
end
end
