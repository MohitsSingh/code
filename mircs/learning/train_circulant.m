function clusters = train_circulant(conf,clusters,neg_images)

% creates once a representation of all negatives windows using fft
% train each of the detectors against this representation.

% new_parameters = struct('sampling',sampling, 'features',features, 'cell_size',cell_size, ...
set_defaults;
set_custom_defaults;
fprintf('collecting neg samples...');
neg_samples = getNegSamples(conf,neg_images,sampling,cell_size);%,features);
fprintf('done!');

%profile of labels for positive samples, for each shift
sz = size(neg_samples);
% pos_labels = gaussian_shaped_labels(target_magnitude, target_sigma, sz);
pos_labels = zeros(sz(1:2));  %labels for all shifted samples
pos_labels(1,1) = target_magnitude;  %label for 0-shift (original sample)
N = sqrt(prod(sz(1:2)));  %constant factor that makes the FFT/IFFT unitary
%transform all data (including training labels) to the Fourier domain
neg_labels = -target_magnitude * ones(sz(1:2));  %same label for all samples

pos_labels = fft2(pos_labels) / N;
neg_labels = fft2(neg_labels) / N;
%set constant frequency of labels to 0 (equivalent to subtracting the mean)
pos_labels(1,1) = 0;
neg_labels(1,1) = 0;
tic
for k = 1:length(clusters)
% % %     fprintf('Training %d out of %d...',k,length(clusters));
    num_pos_samples = size(clusters(k).cluster_samples,2);
    pos_samples = reshape(clusters(k).cluster_samples,sz(1),sz(2),[],num_pos_samples);
    samples = fft2(cat(4,pos_samples,neg_samples)) / N;
    sz = size(samples);
    weights = zeros(sz(1:3));
    bias = 0;
    
    %circulant decomposition (non-parallel code).
    
    y = zeros(sz(4),1);  %sample labels (for a fixed frequency)
    progress Training
    
    for r = 1:sz(1),
        for c = 1:sz(2),
            %fill vector of sample labels for this frequency
            y(:) = neg_labels(r,c);
            y(1:num_pos_samples) = pos_labels(r,c);
            
            %train classifier for this frequency
            weights(r,c,:) = linear_training(training, double(permute(samples(r,c,:,:), [4, 3, 1, 2])), y);
        end
        
        progress(r, sz(1));
    end
            
    %transform solution back to the spatial domain
    weights = real(ifft2(weights)) * N;
    
    
    %crop template by some cells if needed
    crop = floor(cropping_cells / 2);
    weights = weights(1 + crop : end - crop, 1 + crop : end - crop, :);
    clusters(k).w = weights(:);
    assert(~isempty(weights), 'Too much cropping.')
    
%     fprintf('done\n');
    
end

% % % fprintf('finished training %d classifiers in %4.3f seconds\n',length(clusters),toc);



function neg_samples = getNegSamples(conf,neg_images,sampling,cell_size,features)

sample_sz = [conf.features.winsize 32];%*conf.detection.params.init_params.sbin;
%list training image files for all classes
n = numel(neg_images);

%stride size, in cells (vertical and horizontal directions)
stride_sz = floor(sampling.neg_stride * sample_sz(1:2));
%compute max. number of samples given the image size and stride
%(NOTE: this is probably a pessimistic estimate!)
% num_neg_samples = numel(neg_images) * prod(floor(sampling.neg_image_size / cell_size ./ stride_sz))/2;
%initialize data structure for all samples, starting with positives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get the negative samples
%neg_samples = zeros([sample_sz(1:3), num_neg_samples], 'single');
neg_samples = {};

%%samples = cat(4, pos_samples, zeros([sample_sz(1:3), num_neg_samples], 'single'));

progress();


for f = 1:n,
%     f
    %load image and bounding box info
    %[boxes, im] = dataset_image(dataset, class, neg_images{f});
    im = neg_images{f};
    if (ischar(im))
        im = imread(neg_images{f});
    end
    
    
    %ensure maximum size
    if size(im,1) > sampling.neg_image_size(1),
        im = imresize(im, [sampling.neg_image_size(1), NaN], 'bilinear');
    end
    if size(im,2) > sampling.neg_image_size(2),
        im = imresize(im, [NaN, sampling.neg_image_size(2)], 'bilinear');
    end
    
    %extract features (e.g., HOG)
    %x = conf.features.fun(im);
    x = fhog(im2single(im),cell_size);
    %x = get_features(im, features, cell_size);
    
    %extract subwindows, given the specified stride
    rRange = 1 : stride_sz(1) : size(x,1) - sample_sz(1) + 1;
    cRange = 1 : stride_sz(2) : size(x,2) - sample_sz(2) + 1;
    nNegSamples = length(rRange)*length(cRange);
    
    conf.detection.params.detect_min_scale = 1;
    curNegSamples = allFeatures(conf,im,.2);
    curNegSamples = reshape(curNegSamples,sample_sz(1),sample_sz(2),sample_sz(3),[]);
    
    
%     curNegSamples = zeros(sample_sz(1),sample_sz(2),sample_sz(3),nNegSamples);        
%     conf.detection.params.detect_levels_per_octave = 1;
% 
%     idx = 1;
%     for r = rRange,
%         for c = cRange,
%             %store the sample
%             curNegSamples(:,:,:,idx) = x(r : r+sample_sz(1)-1, c : c+sample_sz(2)-1, :);
%             idx = idx + 1;
%         end
%     end
%     
    neg_samples{end+1} = curNegSamples;
    
    progress(f, n);
end

neg_samples = cat(4,neg_samples{:});

%trim any uninitialized samples at the end
% if idx - 1 < size(neg_samples,4),
%     neg_samples(:,:,:, idx : end) = [];
% end

