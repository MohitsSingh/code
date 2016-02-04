classdef AugmentationHelper < handle
    
    properties
        norm_type
        kermap
        prepool_norm_type
        subcode_norm_type
        
        augmentation
        augmentation_collate
        run_mode
        
        conv_to_grey
        c_image
        downsize_to_fixed_dims
        centre_only_crop
    end
    
    properties(SetAccess=protected)
        default_consts_
        run_configs_
    end
    
    methods
        function obj = AugmentationHelper(varargin)
            
            obj.default_consts_ = struct('IMAGE_DIM', 256, ...
                                         'CROPPED_DIM', 224, ...
                                         'AUG_SAMPLEGRID_RE', 'samplegrid_([0-9]+)', ...
                                         'AUG_SPM_RE', 'spm_([0-9]+)');
            obj.default_consts_.DEFAULT_RUN_CONFIG_NAMES = {'train','test'};
            obj.default_consts_.RUN_CONFIG_FIELDS = {'augmentation', 'augmentation_collate'};
            
            p = inputParser();
            % basic options ---------------
            % -----------------------------
            p.addParamValue('norm_type', 'l2', ... % final feature normalisation
                            @(x)any(validatestring(x,{'l2','l1','none'})));
            p.addParamValue('kermap', 'none', ... % kernel map
                            @(x)any(validatestring({'none', 'hellinger'},x)));
            p.addParamValue('prepool_norm_type', 'none', ... % norm prior to pooling
                            @(x)any(validatestring(x,{'l2','l1','none'})));
            p.addParamValue('subcode_norm_type', 'l2', ... % norm of subcodes when stacking
                            @(x)any(validatestring({'l2','l1','none',x})));
            % augmentation ---------------
            % ----------------------------
            % basic usage - just set augmentation options directly
            p.addParamValue('augmentation', 'none', ...
                            @(x)obj.validateRunConfigProperty('augmentation', x));
            p.addParamValue('augmentation_collate', 'none', ...
                            @(x)obj.validateRunConfigProperty('augmentation_collate', x));
            p.addParamValue('run_mode', 'test', @ischar);
            % advanced usage - specify a struct of augmentation options for
            % different named configurations, then choose which option set is
            % active by setting the 'run_mode' property
            p.addParamValue('run_configs', [], ... % optionally specify run configs manually
                            @obj.validateRunConfigs);
            % advanced options ---------------
            % --------------------------------
            p.addParamValue('conv_to_grey', false, @islogical);
            p.addParamValue('c_image', false, @islogical);
            p.addParamValue('downsize_to_fixed_dims', false, @islogical);
            p.addParamValue('centre_only_crop', false, @islogical);
            
            if nargin >= 3
                p.parse(varargin{:});
            else
                p.parse();
            end
            
            parsed_opts = p.Results;
            parsed_opts_names = fieldnames(parsed_opts);
            
            override_run_configs = ~isempty(parsed_opts.run_configs);
            if override_run_configs
                obj.run_configs_ = parsed_opts.run_configs;
            end
            
            % parse output from inputParser into class properties
            for oi = 1:length(parsed_opts_names)
                iter_name = parsed_opts_names{oi};
                cfg_fields = obj.default_consts_.RUN_CONFIG_FIELDS;
                if ~any(ismember(cfg_fields, iter_name))
                    if ~strcmp('run_mode', iter_name) && ~strcmp('run_configs', iter_name)
                        % set property directly
                        obj.(iter_name) = parsed_opts.(iter_name);
                    end
                else
                    % augmentation options belong in a run_config
                    if ~override_run_configs
                        % or set up a set of default run_configs
                        run_config_names = obj.default_consts_.DEFAULT_RUN_CONFIG_NAMES;
                        for fi = 1:length(run_config_names)
                            conf_name = run_config_names{fi};
                            obj.run_configs_.(conf_name).(iter_name) = ...
                                parsed_opts.(iter_name);
                        end
                    end
                end
            end
            
            % finally, set run mode to get everything started
            obj.run_mode = parsed_opts.run_mode;
            
        end
        
        % Public Interface
        images = prepareImage(obj, im, varargin);
        code = transformCodes(obj, codes);
        
        dim = get_output_dim_mul(obj, run_mode);
        count = get_output_count(obj, run_mode);
        
        image_count = get_aug_image_count(obj, aug_type);
        
        
        % Property setter / getter
        
        function set.augmentation(obj, value)
            run_config = obj.get_run_config();
            run_config.augmentation = value;
            obj.set_run_config(run_config);
        end
        
        function value = get.augmentation(obj)
            run_config = obj.get_run_config();
            value = run_config.augmentation;
        end
        
        function set.augmentation_collate(obj, value)
            run_config = obj.get_run_config();
            run_config.augmentation_collate = value;
            obj.set_run_config(run_config);
        end
        
        function value = get.augmentation_collate(obj)
            run_config = obj.get_run_config();
            value = run_config.augmentation_collate;
        end
            
        function set.run_mode(obj, value)
            if ~isempty(obj.run_configs_) && ~isfield(obj.run_configs_, value)
                if isempty(obj.run_mode)
                    obj.run_mode = value;
                    warning('Setting to non-existing config as empty: %s', value);
                else
                    error('There is no config: %s', value);
                end
            else
                obj.run_mode = value;
            end
        end
        
        % Advanced - set custom run configurations
        
        run_config = get_run_config(obj, run_mode)
        set_run_config(obj, run_config, run_mode)
        
    end
    
    % ******************************************
    % Start of protected utility methods
    % ******************************************
    
    methods(Access=protected)
        
        % Run config validation
        function is_valid = validateRunConfigs(obj, run_configs)
            is_valid = true;
            if isempty(run_configs)
                return;
            end
            
            if ~isstruct(run_configs)
                error('run_configs must be struct');
            end
            
            fnames = fieldnames(run_configs);
            for fi = 1:length(fnames)
                obj.validateRunConfig(run_configs.(fnames{fi}));
            end
        end
        
        function validateRunConfig(obj, run_config)
            fnames = fieldnames(run_config);
            for fi = 1:length(fnames)
                obj.validateRunConfigProperty(fnames{fi}, run_config.(fnames{fi}));
            end
        end
        
        function is_valid = validateRunConfigProperty(obj, property_name, value)
            if strcmp(property_name, 'augmentation')
                is_valid = any(validatestring(value,{'none','aspect_corners','centre','flip'}));
            elseif strcmp(property_name, 'augmentation_collate')
                is_valid = any(validatestring(value,{'none','sum','max','stack','sumraw','maxraw'}));
            else
                error('Unknown run_config property: %s', property_name);
            end
        end
    end
    
end
