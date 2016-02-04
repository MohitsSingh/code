function set_run_config(obj, run_config, run_mode)

    obj.validateRunConfig(run_config);

    if nargin < 3 || isempty(run_mode)
        obj.run_configs_.(obj.run_mode) = run_config;
    else
        obj.run_configs_.(run_mode) = run_config;
    end

end
