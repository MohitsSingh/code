
classdef ClassifierFeatureExtractor < FeatureExtractor
    properties
        extractor
        classifier
    end
    methods
        function obj = ClassifierFeatureExtractor(conf,extractor,classifier)
            obj = obj@FeatureExtractor(conf);
            obj.extractor = extractor;
            obj.classifier = classifier;
        end
        function x = extractFeaturesHelper(obj,imageID, roi)
            x = obj.extractor.extractFeatures(imageID,roi);
            [ignore_ x] = obj.classifier.test(x);
        end
        
        function x = description(obj)
            x = [obj.extractor.description() '_classifier'];
        end
    end    
end