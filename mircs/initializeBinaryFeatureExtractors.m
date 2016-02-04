function extractors = initializeBinaryFeatureExtractors(conf)
    extractors{1} = RelativeFeatureExtractor(conf)
end