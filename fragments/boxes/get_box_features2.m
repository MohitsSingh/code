function [ features ] = get_box_features2(globalOpts,sample_boxes,sample_image_ids,model,scale)
%get_box_features summary of this function goes here
%   detailed explanation goes here

nSamples = size(sample_boxes,1);
perc_jump = 5;
next_perc = 0;

dummy_hist = buildSpatialHist2(zeros(5),...
    [1 1 3 3], globalOpts);

hist_size = length(dummy_hist);

if (globalOpts.useGistFeatures)
    features = zeros(960,nSamples);
else
    features = zeros(hist_size,nSamples,'single');
end
prev_id = [];
prev_quantPath = [];

orientationsPerScale = [8 8 4];
numberBlocks = 4;

% G = createGabor2(orientationsPerScale, 32, 32);

for k = 1:nSamples
    %     k
    current_perc = 100*k/nSamples;
    %     fprintf('extracting histogram for sample %d/%d [%s%03.3f]\n', ...
    %         k, nSamples, '%', current_perc)
    %
    if (current_perc >= next_perc)
        fprintf('extracting histogram for sample %d/%d [%s%03.3f]\n', ...
            k, nSamples, '%', current_perc)
        next_perc = current_perc + perc_jump;
    end
    
    bbox = sample_boxes(k,:);
    
    if (length(sample_image_ids) == 1)
        currentID = sample_image_ids{1};
    else
        currentID = sample_image_ids{k};
    end
    if (~strcmp(currentID,prev_id))
        prev_id = currentID;
        im = imread(getImageFile(globalOpts,currentID));
    end
    %
    
    if (globalOpts.useGistFeatures)
        sub_im = im(bbox(1):bbox(3),bbox(2):bbox(4),:);
        sub_im = imresize(sub_im,[32 32]);
        output = prefilt(double(sub_im), 4);
        features(:,k) = gistGabor(output, numberBlocks, G);
        
    elseif (globalOpts.det_rescale == 0)
        
        quantPath = getQuantFile(globalOpts,[currentID '_' num2str(scale)]);
        quantPath_old = getQuantFile(globalOpts,[currentID]);
        
        if (scale == 1 && exist(quantPath_old,'file'))
            quantPath = quantPath_old;
        end
        
        if (~strcmp(prev_quantPath,quantPath))
            if (~exist(quantPath,'file'))
                warning(['gettraininginstances ---> quantized descriptors for image' currentID ' don''t exist']);
                im = imresize(im,scale);
                [F,D] = globalOpts.descfun(im,globalOpts.phowOpts{:});
                quantized = double(vl_kdtreequery(model.kdtree, model.vocab, ...
                    D, ...
                    'MaxComparisons', 15)) ; %#ok<NASGU>
                
                save(quantPath,'quantized','F');
            else
                
                quantImagePath = strrep(quantPath,'.mat',...
                    ['_' num2str(globalOpts.scale_choice) '_img.mat']);
                                                
                if (exist(quantImagePath,'file'))
                    load(quantImagePath);
                else
                    
                    load(quantPath);
                    [quantImage] = im2QuantImage( F,quantized,[size(im,1),size(im,2)],globalOpts);
                    save(quantImagePath,'quantImage');
                end
                if (isfield(globalOpts,'map'))
                    q = quantImage(quantImage~=0);
                    q = globalOpts.map(q);
                    quantImage(quantImage~=0) = q;
                    quantImage(quantImage==0) = globalOpts.numWords+1;
                end
                %                 k
                %                 imagesc(quantImage);
                %                 pause;
            end
            prev_quantPath = quantPath;
        end
        
        bbox = bbox*scale;
        
        
        use_old_method = 0;                
        hist_ = buildSpatialHist2(quantImage,bbox([2 1 4 3]),globalOpts);        
        features(:,k) = hist_;
    else % here, we first normalized each box so that it's of
        % the same scale and only then extract features and quantize
        % them
                
        [~,~,bArea] = BoxSize(bbox);
        m_ = im;
        sz_ratio = (globalOpts.det_rescale/(bArea^.5));
        if (sz_ratio < 1) % don't up-size images...
            %             sub_im = imresize(sub_im,sz_ratio);
            m_ = imresize(m_,sz_ratio);
            bbox = round(bbox*sz_ratio);
        end
        
        sub_im = m_(max(1,bbox(1)-9):min(size(m_,1),bbox(3)+9),...
            max(1,bbox(2)-9):min(size(m_,2),bbox(4)+9));
        
        phowOpts = {'Step', 1, 'Sizes', [4],'Fast',1,'FloatDescriptors',1};
        [F,D] = globalOpts.descfun(sub_im,phowOpts{:});
        fff = F(4,:);
        fff = fff==globalOpts.scale_choice;
        F = F(:,fff);
        D = D(:,fff);
        quantized = double(vl_kdtreequery(model.kdtree, model.vocab, ...
            D, 'maxcomparisons', 15));
        
        [quantImage] = im2QuantImage( F,quantized,[size(sub_im,1),size(sub_im,2)],globalOpts);
        
        features(:,k) = buildSpatialHist2(quantImage,...
            [1 1 size(quantImage,2) size(quantImage,1)],...
            globalOpts);
    end
end
disp('finished extracting histograms...');

end

