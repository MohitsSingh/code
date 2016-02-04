function [partFeats,linkFeats,shapeFeats,localPatches]  = getPartFeats(I,prevPart,nextParts,featureExtractor,params)
linkFeats = [];
shapeFeats = [];
if ~strcmp(params.feature_extraction_mode,'bbox')
    error('currently not supporting e.g, masked feature extraction')
end

localPatches = {};
localShapes = {};
hasPrev = ~isempty(prevPart);
if hasPrev
    prevBox = prevPart.bbox;
    connectionPatches = {};
end
fillValue = .5;
if isa(I,'uint8')
    fillValue = 128;
end

if strcmp(params.cand_mode,'boxes')    
    for t = 1:length(nextParts)
        curBox = round(nextParts(t).bbox);
        localPatches{t} = cropper(I,curBox);
        if (params.interaction_features && hasPrev)
            box_int = getInteractionRegion(prevBox,curBox);
            connectionPatches{end+1} = cropper(I,box_int);
        end
    end
elseif strcmp(params.cand_mode,'polygons')
    for t = 1:length(nextParts)
        curPart = nextParts(t);
        curBox = pts2Box(curPart.xy);
        curBox = round(makeSquare(inflatebbox(curBox,1.2,'both',false),true));
        curPoly = bsxfun(@minus,curPart.xy,curBox(1:2));
        [a,b] = BoxSize(curBox);        
        curPatch = cropper(I,curBox);
        z = poly2mask2(curPoly,size2(curPatch));
        z = repmat(z,[1 1 size(curPatch,3)]);
        curPatch(~z) = fillValue;
        localPatches{end+1} = curPatch;
    end
else    
    for t = 1:length(nextParts)
        curPart = nextParts(t);
        localPatches{end+1} = maskedPatch(I,curPart.mask,true);
        Z = im2double(repmat(curPart.mask,[1 1 3]));
        localShapes{end+1} = maskedPatch(Z,curPart.mask,true,0);
    end
end

partFeats = featureExtractor.extractFeaturesMulti(localPatches,false);
if ~isempty(localShapes)
    shapeFeats = featureExtractor.extractFeaturesMulti(localShapes,false);
end


if (params.interaction_features && hasPrev && ~isempty(connectionPatches))
    if strcmp(params.cand_mode,'polygons');
        error 'polygonal interaction features not supported for now'
    end
    linkFeats = featureExtractor.extractFeaturesMulti(connectionPatches,false);
end
end