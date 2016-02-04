classdef FeatureExtractorBOW < FeatureExtractor
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dict
    end
    
    methods
        function obj = FeatureExtractorBOW(dictPath,featParams)
            load(dictPath);
            obj.dict = dict;
        end
end

