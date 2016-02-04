function images = prepareImage(obj, im, varargin)
    
    p = inputParser();
    p.addParamValue('mean_img', []);
    p.addParamValue('preproc_dup_grey', false, @islogical);
    p.addParamValue('output_as_cell', false, @islogical);
    p.addParamValue('fixed_prep_image', true, @islogical);
    if nargin >= 3
        p.parse(varargin{:});
    else
        p.parse();
    end
    opts = p.Results;
    
    assert(isa(im, 'single'));
    assert(ndims(im) == 3);
    
    run_config = obj.get_run_config();
    
    images_cell = {};
    
    % 1. prepare parameters
    
    if obj.downsize_to_fixed_dims
        IMAGE_DIM = obj.default_consts_.IMAGE_DIM;
        CROPPED_DIM = obj.default_consts_.CROPPED_DIM;
    else
        IMAGE_DIM = min([size(im, 1), size(im, 2)]);
        CROPPED_DIM = floor(0.875*IMAGE_DIM);
    end
    
    if ~obj.centre_only_crop && ...
            ~strcmp(run_config.augmentation, 'aspect_corners')
        if obj.downsize_to_fixed_dims
            IMAGE_DIM = CROPPED_DIM;
        else
            CROPPED_DIM = IMAGE_DIM;
        end
    end
    
    if obj.c_image
        DIM_ORDER = [2 1 3];
    else
        DIM_ORDER = [1 2 3];
    end
    FLIP_DIM = find(DIM_ORDER == 2);
    
    % 2a. prepare base image
    
    if opts.preproc_dup_grey
        im = dup_grey_im(im);
    end
    
    if obj.downsize_to_fixed_dims
        im = downsizeImage(im, IMAGE_DIM);
    end
    
    % 2b. prepare mean image
    
    IMAGE_MEAN = 0;
    
    if ~isempty(opts.mean_img)
        
        if ~obj.downsize_to_fixed_dims
            error('downsize_to_fixed_dims must be true if mean image is specified');
        end
        
        if ischar(opts.mean_img)
            d = load(opts.mean_img);
            mean_img = d.mean_img;
        else
            mean_img = opts.mean_img;
        end
        
        if false
            if size(mean_img, 1) ~= IMAGE_DIM && ...
                    size(mean_img, 2) ~= IMAGE_DIM
                mean_img = downsizeImage(mean_img, IMAGE_DIM);
            end
        end
        
        IMAGE_MEAN = centre_crop(mean_img, CROPPED_DIM);
        
        if opts.preproc_dup_grey
            IMAGE_MEAN = dup_grey_im(IMAGE_MEAN);
        end
    end
    
    % 3. image extraction
    
    grid_aug = ~isempty(regexp(run_config.augmentation, ...
                               obj.default_consts_.AUG_SAMPLEGRID_RE, 'once'));
    
    if strcmp(run_config.augmentation, 'aspect_corners') || grid_aug
        
        if grid_aug
            grid_token = regexp(run_config.augmentation, ...
                                obj.default_consts_.AUG_SAMPLEGRID_RE, 'tokens');
            grid_token = grid_token{1};
            centre_grid_dim = str2double(grid_token{1});
            clear grid_token;
        else
            centre_grid_dim = 1;
        end
        
        [indices_i, indices_j, ac_image_count] = ...
            prepareEvenGridIndices(centre_grid_dim+2, ...
                                   im, CROPPED_DIM);
        
        images = zeros(CROPPED_DIM, CROPPED_DIM, 3, ac_image_count*2, ...
                       'single');
        
        curr = 1;
        for i = indices_i
            for j = indices_j
                if xor(i == 1 || i == indices_i(2), ...
                       j == 1 || j == indices_j(2))
                    continue;
                end
                
                if opts.fixed_prep_image
                    images(:, :, :, curr) = ...
                        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :) - IMAGE_MEAN, DIM_ORDER);
                else
                    images(:, :, :, curr) = ...
                        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), DIM_ORDER) - ...
                        IMAGE_MEAN;
                end
                images(:,:,:,curr+ac_image_count) = flipdim(images(:, :, :, curr), FLIP_DIM);
                curr = curr + 1;
            end
        end
        
        assert(curr == ac_image_count+1);
        
    else
        switch run_config.augmentation
          case 'flip'
            if obj.downsize_to_fixed_dims || obj.centre_only_crop
                images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 2, 'single');
                if opts.fixed_prep_image
                    images(:,:,:,1) = ...
                        permute(centre_crop(im, CROPPED_DIM) - IMAGE_MEAN, DIM_ORDER);
                else
                    images(:,:,:,1) = ...
                        permute(centre_crop(im, CROPPED_DIM), DIM_ORDER) - IMAGE_MEAN;
                end
                images(:,:,:,2) = flipdim(images(:,:,:,1), FLIP_DIM);
            else
                % if using flip augmentation with original dims, must output
                % images as a cell array due to varying image dimensions
                if ~opts.output_as_cell
                    error(['Must set output_as_cell to true when ' ...
                           'downsize_to_fixed_dims = 0 and centre_only_crop = 0']);
                end
                images_cell = cell(2);
                images_cell{1} = permute(im, DIM_ORDER);
                images_cell{2} = flipdim(images_cell{1}, FLIP_DIM);
            end
          case 'none'
            
            if obj.downsize_to_fixed_dims || obj.centre_only_crop
                images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 'single');
                
                if opts.fixed_prep_image
                    images(:,:,:,1) = ...
                        permute(centre_crop(im, CROPPED_DIM) - IMAGE_MEAN, DIM_ORDER);
                else
                    images(:,:,:,1) = ...
                        permute(centre_crop(im, CROPPED_DIM), DIM_ORDER) - IMAGE_MEAN;
                end
            else
                images = permute(im, DIM_ORDER);
            end
          otherwise
            error('Unrecognised augmentation type: %s', run_config.augmentation);
        end
    end
    
    if opts.output_as_cell
        
        if isempty(images_cell)
            images_cell = cell(size(images, 4));
            for i = 1:length(images_cell)
                images_cell{i} = images(:,:,:,i);
            end
        end
        
        images = images_cell;
        
    end
    
% -----------------------------------------------------------------------------
    
    function im = downsizeImage(im, smallest_dim)
        
        % resize to IMAGE_DIM x N where IMAGE_DIM is the smaller dimension
        if size(im, 1) < size(im, 2)
            if opts.fixed_prep_image
                im = imresize(im, [smallest_dim NaN]);
            else
                im = imresize(im, [smallest_dim NaN], 'bilinear');
            end
        else
            if opts.fixed_prep_image
                im = imresize(im, [NaN smallest_dim]);
            else
                im = imresize(im, [NaN smallest_dim], 'bilinear');
            end
        end
        
    end
    
    function [indices_i, indices_j, ac_image_count] = ...
            prepareEvenGridIndices(N, im, crop_dim)
        
        max_i = size(im,1) - crop_dim + 1;
        max_j = size(im,2) - crop_dim + 1;
        
        step_i = floor(max_i/(N-1));
        step_j = floor(max_j/(N-1));
        
        indices_i = [1 max_i (step_i+1):step_i:(max_i-step_i+2)];
        indices_j = [1 max_j (step_j+1):step_j:(max_j-step_j+2)];
        assert(length(indices_i) == length(indices_j));
        assert(length(indices_i) == N);
        
        ac_image_count = (N-2)*(N-2) + 4;
        
    end
    
    function cropped_im = centre_crop(im, crop_dim)
        
        excess_sz_i = (size(im,1) - crop_dim)/2;
        excess_sz_j = (size(im,2) - crop_dim)/2;
        
        if excess_sz_i > 0 || excess_sz_j > 0
            excess_sz_i = floor(excess_sz_i) + 1;
            excess_sz_j = floor(excess_sz_j) + 1;
            
            cropped_im = im(excess_sz_i:excess_sz_i+crop_dim-1, ...
                            excess_sz_j:excess_sz_j+crop_dim-1, :);
        else
            cropped_im = im;
        end
        
        assert(size(cropped_im,1) == crop_dim);
        assert(size(cropped_im,2) == crop_dim);
        
    end
    
    function dup_grey_im = dup_grey_im(im)
        if max(im(:)) > 1.0
            im = im/255.0;
            assert(max(im(:)) <= 1.0);
            im = rgb2gray(im);
            im = im*255.0;
        else
            im = rgb2gray(im);
        end
        
        dup_grey_im = repmat(im, [1, 1, 3]);
    end
 
end