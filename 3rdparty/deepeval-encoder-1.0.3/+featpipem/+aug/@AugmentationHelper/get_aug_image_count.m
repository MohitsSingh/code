function image_count = get_aug_image_count(obj, aug_type)
    image_count = [];
    
    if strcmp(aug_type, 'none')
        error('No augmentation!');
    elseif strcmp(aug_type, 'centre_corners') || ...
            strcmp(aug_type, 'aspect_corners') || ...
            strcmp(aug_type, 'squashed_corners')
        centre_grid_dim = 1;
    elseif ~isempty(regexp(aug_type, obj.default_consts_.AUG_SAMPLEGRID_RE, 'once'))
        grid_token = regexp(aug_type, ...
                            obj.default_consts_.AUG_SAMPLEGRID_RE, 'tokens');
        grid_token = grid_token{1};
        centre_grid_dim = str2double(grid_token{1});
    elseif strcmp(aug_type, 'flip')
        image_count = 2;
    elseif ~isempty(regexp(aug_type, obj.default_consts_.AUG_SPM_RE, 'once'))
        grid_token = regexp(aug_type, ...
                            obh.default_consts_.AUG_SPM_RE, 'tokens');
        grid_token = grid_token{1};
        level_count = str2double(grid_token{1});
        assert(level_count > 0);
        
        image_count = 1;
        for i = 1:length(level_count)
            image_count = image_count + (2^i)^2;
        end
    else
        error('Unrecognized augmentation type: %s', aug_type);
    end
    
    if isempty(image_count)
        image_count = (centre_grid_dim*centre_grid_dim + 4)*2;
    end
end