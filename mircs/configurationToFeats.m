function [all_feats]  = configurationToFeats(I,boxes,routes,featureExtractor)
% a configuration is a cell-array describing a chain of image elements.
% extract features from this configuration.
% first, just extract all features and concatenate them.
all_feats = {};

box_inds = unique(routes(:));
feature_cache = cell(size(box_inds));
boxes = double(round(boxes));
nRoutes = size(routes,1);
for iConfig = 1:nRoutes
    iConfig/nRoutes
    local_feats = {};
    connection_feats = {};
    %m = double(round(configuration{iConfig}))
    m = boxes(routes(iConfig,:),:);
    %bb = cellfun2(@(x) x.bbox,m);
    localPatches = {};
    connectionPatches = {};
    isBad = false;
    for u = 1:size(m,1)
        bb = round(m(u,:));
        curBoxInd = routes(iConfig,u);
        
        localPatches{u} = cropper(I,bb); 
        
        if u > 1 % sample inter-node appearance
            poly1 = box2Pts(m(u-1,:));
            poly2 = box2Pts(m(u,:));
            sz = size2(I);
            r1 = poly2mask2(round(poly1),sz);
            r2 = poly2mask2(round(poly2),sz);            
            z = bwdist(r1)+bwdist(r2);
            [u,iu] = min(z(:));
            [yy,xx] = ind2sub(size(z),iu);
            boundaryRegion = round(inflatebbox([xx yy],16,'both',true));
            boundaryRegion = box2Region(boundaryRegion,size2(I));
            connectionPatches{end+1} = cropper(I,region2Box(boundaryRegion));
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