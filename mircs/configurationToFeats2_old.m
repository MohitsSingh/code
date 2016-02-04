function [part_feats,int_feats]  = configurationToFeats2(I,configs,featureExtractor,params)
% a configuration is a cell-array describing a chain of image elements.
% extract features from this configuration.
% first, just extract all features and concatenate them.
part_feats = {};
int_feats = {};
if ~strcmp(params.feature_extraction_mode,'bbox')
    error('currently not supporting e.g, masked feature extraction')
end
ticID = ticStatus('extracting configuration features',.1,.01,true);
for iConfig = 1:length(configs)
    %     iConfig/length(configs)
    local_feats = {};
    connection_feats = {};
    localPatches = {};
    connectionPatches = {};
    isBad = false;
    m = configs{iConfig};
    for u = 1:length(m)
        bb = round(m(u).bbox);
        %         bb(4)-bb(2)
        localPatches{u} = cropper(I,bb);
        if (params.interaction_features)
            if u > 1 % sample inter-node appearance
                poly1 = m(u-1).xy;
                poly2 = m(u).xy;
                sz = size2(I);
                r1 = poly2mask2(round(poly1),sz);
                r2 = poly2mask2(round(poly2),sz);
                z = bwdist(r1)+bwdist(r2);
                [u,iu] = min(z(:));
                [yy,xx] = ind2sub(size(z),iu);
                
                curScale = (sqrt(nnz(r1))+sqrt(nnz(r2)))/2;
                boundaryRegion = round(inflatebbox([xx yy],curScale,'both',true));
                boundaryRegion = box2Region(boundaryRegion,size2(I));
                connectionPatches{end+1} = cropper(I,region2Box(boundaryRegion));
            end
        end
    end
    if (isBad)
        continue
    end
    
    localFeats = featureExtractor.extractFeaturesMulti(localPatches,false);
    localFeats = reshape(localFeats,[],length(localPatches));
    part_feats{end+1} = localFeats;
    if (params.interaction_features)
        connectionFeats = featureExtractor.extractFeaturesMulti(connectionPatches,false);
        connectionFeats = reshape(connectionFeats,[],length(connectionPatches));
        
        %     intra_feats{iConfig} = curFeats(:);
        int_feats{end+1} = connectionFeats;
        %         all_feats{end+1} = [cat(1,connection_feats{:});localFeats(:)];
        %     else
        %         all_feats{end+1} = localFeats(:);
        %     end
        tocStatus(ticID,iConfig/length(configs))
    end
    % fprintf('\n');
end