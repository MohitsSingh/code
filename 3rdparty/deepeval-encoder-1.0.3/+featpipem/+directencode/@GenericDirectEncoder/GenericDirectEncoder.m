classdef GenericDirectEncoder < handle
    %GENERICDIRECTENCODER Generic interface to direct encoder
    
    methods(Abstract)
        dim = get_output_dim(obj)
        count = get_output_count(obj)
        
        set_run_mode(obj, run_mode)
        
        code = encode(obj, im)
    end
    
end

