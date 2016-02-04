function [part_feats,int_feats,shape_feats]  = configurationToFeats2(I,configs,featureExtractor,params)
% a configuration is a cell-array describing a chain of image elements.
% extract features from this configuration.
% first, just extract all features and concatenate them.
part_feats = {};
int_feats = {};
if ~strcmp(params.feature_extraction_mode,'bbox')
    error('currently not supporting e.g, masked feature extraction')
end
ticID = ticStatus('extracting configuration features',.1,.01,true);
% empties = true(3,1);
for iConfig = 1:length(configs)
    %     iConfig/length(configs)
    
    m = configs{iConfig};
    prev_part = [];
    
    curPartFeats = {};
    curIntFeats = {};
    curShapeFeats = {};
    for u = 1:length(m)
        curPart = m(u);
        [curPartFeats{u},curIntFeats{u},curShapeFeats{u}] = getPartFeats(I,prev_part,curPart,featureExtractor,params);        
        prev_part = curPart;
    end
    part_feats{iConfig} = cat(2,curPartFeats{:});
    int_feats{iConfig} = cat(2,curIntFeats{:});
    shape_feats{iConfig} = cat(2,curShapeFeats{:});
    
    tocStatus(ticID,iConfig/length(configs))
end
