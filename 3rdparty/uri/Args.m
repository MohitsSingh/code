classdef Args
    
    properties
        args
    end
    
    methods 

        function obj = Args(args)
            obj.args = struct(args{:});
        end
        
        function out = get(obj, argName, defaultValue)
            
            if isfield(obj.args, argName)
                out = obj.args.(argName);
            else
                out = defaultValue;
            end

        end
    end
    
   
end