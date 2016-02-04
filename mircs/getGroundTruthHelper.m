function [gtMask,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox)
% assume that imgdat has :I,mouthBox
% I = imgData.I;
% mouthBox = imgData.mouthBox;
gtMask = [];
nodes = params.nodes;
gt_graph = get_gt_graph(imgData,nodes,params,I);
% 2. remove "trivial" regions
isValid = false;
if ~isfield(gt_graph{2},'roiMask')
    return
end
m = gt_graph{2}.roiMask;
if isempty(m) || nnz(m)==0
    return
end
gtMask = cropper(m,mouthBox);
if (nnz(gtMask)==0)
    return
end

%mask should contain at least 15 pixels to be valid.
if nnz(gtMask) < 15
    return;
end

isValid = true;
