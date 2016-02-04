function run_config = get_run_config(obj, run_mode)

    if nargin < 2 || isempty(run_mode)
        run_config = obj.run_configs_.(obj.run_mode);
    else
        run_config = obj.run_configs_.(run_mode);
    end

end
