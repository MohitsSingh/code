function count = get_output_count(obj, run_mode)

    if nargin < 2 || isempty(run_mode)
        run_config = obj.get_run_config();
    else
        run_config = obj.get_run_config(run_mode);
    end

    if ~strcmp(run_config.augmentation, 'none') && ...
            strcmp(run_config.augmentation_collate, 'none')
        count = obj.get_aug_image_count(run_config.augmentation);
    else
        count = 1;
    end

end
