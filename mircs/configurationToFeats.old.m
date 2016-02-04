function [all_feats]  = configurationToFeats(I,configuration,nodes,featureExtractor)
% a configuration is a cell-array describing a chain of image elements.
% extract features from this configuration.
% first, just extract all features and concatenate them.
all_feats = {};
for iConfig = 1:length(configuration)
%     iConfig/length(configuration)
    local_feats = {};
    connection_feats = {};
    m = configuration{iConfig};
    bb = cellfun2(@(x) x.bbox,m);
    localPatches = {};
    connectionPatches = {};
    isBad = false;
    for u = 1:length(m)
        bb = round(m{u}.bbox);
        if (isempty(bb))
            isBad = true;
            break;
        end
        if strcmp(nodes(u).type,'poly')
            if (isempty(m{u}.poly))
                warning('got empty polygon when attempting to extract features');
                isBad = true;
                local_feats{iConfig} = [];
                break;
            end
            localPatches{u} = rectifyWindow(I,m{u}.poly,[64 64]);
        else
            localPatches{u} = cropper(I,bb);
        end
        
        if u > 1 % sample inter-node appearance
            poly1 = m{u-1}.poly;
            poly2 = m{u}.poly;
            sz = size2(I);
            r1 = poly2mask2(round(poly1),sz);
            r2 = poly2mask2(round(poly2),sz);
            
            z = bwdist(r1)+bwdist(r2);
            [u,iu] = min(z(:));
            [yy,xx] = ind2sub(size(z),iu);
            boundaryRegion = round(inflatebbox([xx yy],16,'both',true));
            boundaryRegion = box2Region(boundaryRegion,size2(I));
%             d1 = bwdist(r1) <= 2;
%             d2 = bwdist(r2) <= 2;            
            % find the boundary, which is < 1 far from both groups
%             boundaryRegion = d1 & d2;
            connectionPatches{end+1} = cropper(I,region2Box(boundaryRegion));
            %
%             connection_feats{end+1} = x(:);
        end
    end
    if (isBad)
        continue
    end
        
    localFeats = featureExtractor.extractFeaturesMulti(localPatches,false);
    localFeats = reshape(localFeats,[],length(localPatches));
    connectionFeats = featureExtractor.extractFeaturesMulti(connectionPatches,false);
    connectionFeats = reshape(connectionFeats,[],length(connectionPatches));
    
%     intra_feats{iConfig} = curFeats(:);
    all_feats{end+1} = [cat(1,connection_feats{:});localFeats(:)];
    
end
end