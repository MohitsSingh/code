function best_configuration = decode_structure(adj,unary_scores,edgePot)

adj = full(adj)>0;
% adj = adj | adj';

nStates = size(unary_scores,1);
[edgeStruct] = UGM_makeEdgeStruct(adj,nStates);
nodePot = unary_scores';
for t = 1:size(edgePot,3)
    edgePot(:,:,t) = setdiag(edgePot(:,:,t),0);
end
%[best_configuration] = UGM_Decode_Tree(nodePot,edgePot,edgeStruct);
[best_configuration] = UGM_Decode_TRBP(nodePot,edgePot,edgeStruct);

% [best_configuration] = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);