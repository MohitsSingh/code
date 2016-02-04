classdef classifier < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        conf;
        initialized = false;
        w
        b
    end
    
    methods
        function obj = classifier(conf)
            obj.conf = conf;
        end
        function pred = classify(obj,x)
            pred = obj.w'*x-obj.b;
        end        
    end
    
    methods (Abstract)
        x = getFeatures(obj,imageID,roi);    
                
    end
end
    
