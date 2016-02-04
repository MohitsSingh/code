classdef ConvNetEncoder < handle & featpipem.directencode.GenericDirectEncoder
    %CONVNETENCODER Direct encoding using ConvNets

    properties
        param_file
        model_file
        average_image

        output_blob_name
        preproc_dup_grey
    end
    
    properties(Dependent)
        augmentation
        augmentation_collate
        run_mode
        
        norm_type
        centre_only_for_single_im
    end

    properties(SetAccess=protected)
        caffe_version
        backend
        augmentation_helper_
    end

    properties(Transient, Access=protected)
        net_handle_ = 0
    end
    
    methods(Static)
        function set_backend(backend, device_id)
            if nargin < 2
                device_id = 0;
            end
            
            switch backend
              case 'cpu'
                if nargin >= 2
                    error('Can''t set device_id for CPU backend');
                end
                caffe('set_mode_cpu');
              case 'cuda'
                caffe('set_mode_cuda', device_id);
              otherwise
                error('Unrecognised mode: %s', backend);
            end
        end
    end

    methods
        function obj = ConvNetEncoder(varargin)

            p = inputParser();
            p.addRequired('param_file', @ischar);
            p.addRequired('model_file', @ischar);
            p.addRequired('average_image', @ischar);

            p.addParamValue('output_blob_name', 'fc7', @ischar);
            p.addParamValue('preproc_dup_grey', false, @islogical);

            p.addParamValue('augmentation', 'none', ...
                            @(x)any(validatestring(x,{'none','aspect_corners','centre'})));
            p.addParamValue('augmentation_collate', 'none', ...
                            @(x)any(validatestring(x,{'none','sum','max'})));
            p.addParamValue('run_mode', 'test', @ischar);
            p.addParamValue('run_configs', []); % (advanced)

            p.addParamValue('norm_type', 'l2', ...
                            @(x)any(validatestring(x,{'l2','l1','none'})));
            p.addParamValue('centre_only_for_single_im', false, @islogical);

            p.addParamValue('caffe_version', 1.1, @isnumeric); % max version = 1.1

            p.parse(varargin{:});

            parsed_opts = p.Results;
            parsed_opts_names = fieldnames(parsed_opts);

            % parse output from inputParser into class properties
            for oi = 1:length(parsed_opts_names)
                iter_name = parsed_opts_names{oi};
                if ~any(ismember({'augmentation', 'augmentation_collate', ...
                                  'run_mode', 'run_configs'}, iter_name))
                    obj.(iter_name) = parsed_opts.(iter_name);
                end
            end

            % setup augmentation helper
            aug_helper_opts = ...
                {... % fixed options
                 'c_image', true, ...
                 'downsize_to_fixed_dims', true, ...
                 ... % options passed from ConvNetEncoder class
                 'augmentation', parsed_opts.augmentation, ...
                 'augmentation_collate', parsed_opts.augmentation_collate, ...
                 'run_mode', parsed_opts.run_mode, ...
                 'run_configs', parsed_opts.run_configs, ...
                 'norm_type', parsed_opts.norm_type, ...
                 'centre_only_crop', parsed_opts.centre_only_for_single_im};
            obj.augmentation_helper_ = ...
                featpipem.aug.AugmentationHelper(aug_helper_opts{:});

            % setup net
            obj.initNet_()
        end
        
        function delete(obj)
            try
                caffe('destroy_net_by_handle', obj.net_handle_);
            catch
            end
        end

        % --------------------------------------------------------------

        function dim = get_output_dim(obj, run_mode)
            if nargin < 2, run_mode = []; end
            dim = obj.get_net_output_dim_();
            dim = dim*obj.augmentation_helper_.get_output_dim_mul(run_mode);
        end

        function count = get_output_count(obj, run_mode)
            if nargin < 2, run_mode = []; end
            count = obj.augmentation_helper_.get_output_count(run_mode);
        end

        code = encode(obj, im)

        % --------------------------------------------------------------

        function set.param_file(obj, value)
            obj.param_file = value;
            if obj.net_handle_ > 0, obj.initNet_(); end
        end

        function set.model_file(obj, value)
            obj.model_file = value;
            if obj.net_handle_ > 0, obj.initNet_(); end
        end

        %--
        function set.backend(obj, value)
            error('backend is readonly');
        end
        function value = get.backend(obj)
            value = caffe('get_backend');
        end
        %--

        function set.augmentation(obj, value)
            obj.augmentation_helper_.augmentation = value;
            if obj.net_handle_ > 0, obj.initNet_(); end
        end

        function value = get.augmentation(obj)
            value = obj.augmentation_helper_.augmentation;
        end
        %--
        function set.augmentation_collate(obj, value)
            obj.augmentation_helper_.augmentation_collate = value;
        end

        function value = get.augmentation_collate(obj)
            value = obj.augmentation_helper_.augmentation_collate;
        end
        %--
        function set.run_mode(obj, value)
            obj.augmentation_helper_.run_mode = value;
            if obj.net_handle_ > 0, obj.initNet_(); end
        end

        function value = get.run_mode(obj)
            value = obj.augmentation_helper_.run_mode;
        end
        %--
        function set.norm_type(obj, value)
            obj.augmentation_helper_.norm_type = value;
        end

        function value = get.norm_type(obj)
            value = obj.augmentation_helper_.norm_type;
        end
        %--
        function set.centre_only_for_single_im(obj, value)
            obj.augmentation_helper_.centre_only_crop = value;
        end

        function value = get.centre_only_for_single_im(obj)
            value = obj.augmentation_helper_.centre_only_crop;
        end

        % --------------------------------------------------------------

        function run_config = get_run_config(obj, run_mode)
            if nargin < 2
                run_mode = [];
            end

            run_config = obj.augmentation_helper_.get_run_config(run_mode);
        end

        function set_run_config(obj, run_config, run_mode)
            if nargin < 3
                run_mode = [];
            end

            obj.augmentation_helper_.set_run_config(run_config, run_mode);

            obj.initNet_();
        end

        function set_run_mode(obj, run_mode)
            obj.run_mode = run_mode;
        end

    end

    methods(Access=protected)

        function initNet_(obj)
            run_config = obj.augmentation_helper_.get_run_config();

            if strcmp(run_config.augmentation, 'none')
                input_dim = 1;
            else
                input_dim = ...
                    obj.augmentation_helper_.get_aug_image_count(run_config.augmentation);
            end

            if obj.net_handle_ == 0
                fprintf('Initializing caffe network...\n');
                obj.net_handle_ = caffe('init', obj.param_file, obj.model_file, input_dim, -1);
                %fprintf('Initialized with handle: %d\n', obj.net_handle_);
            else
                fprintf('Re-initializing caffe network...\n');
                assert(obj.net_handle_ > 0);
                caffe('init', obj.param_file, obj.model_file, input_dim, obj.net_handle_);
            end
        end

        function dim = get_net_output_dim_(obj)
            assert(obj.net_handle_ > 0);

            dim = caffe('get_output_dim', obj.output_blob_name, obj.net_handle_);
            dim = dim{1};
        end
    end

    methods(Static=true)
        function obj = loadobj(a)
            fprintf('Loading ConvNetEncoder from file...\n');
            obj = featpipem.directencode.ConvNetEncoder(a.param_file, ...
                                                        a.model_file, ...
                                                        a.average_image, ...
                                                        'caffe_version', a.caffe_version);
            obj.output_blob_name = a.output_blob_name;
            obj.preproc_dup_grey = a.preproc_dup_grey;
            obj.augmentation_helper_ = a.augmentation_helper_;

            obj.initNet_();
        end
    end

end
