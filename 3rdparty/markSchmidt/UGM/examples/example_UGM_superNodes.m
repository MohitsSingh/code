clear all
close all

nSeats = 6;
nRows = 40;
nNodes = nSeats*nRows;
nStates = 2;
adj = latticeAdjMatrix(nSeats,nRows);
edgeStruct = UGM_makeEdgeStruct(adj,nStates);

if 0
	% Drawing the graph
	s = 1;
	for r = 1:nRows
		fprintf('(%3d)-(%3d)-(%3d)-(%3d)-(%3d)-(%3d)\n',s,s+1,s+2,s+3,s+4,s+5);
		s = s+nSeats;
		
		if r < nRows
			fprintf('  |     |     |     |     |     |\n');
		end
	end
end

alpha = 1;
beta = 3;
nodePot = [ones(nNodes,1) alpha*ones(nNodes,1)];
edgePot = repmat([beta 1;1 beta],[1 1 edgeStruct.nEdges]);

superNodes = cell(nRows,1);
for r = 1:nRows
	superNodes{r} = 1 + (r-1)*nSeats:nSeats + (r-1)*nSeats;
end

optimalDecoding = UGM_Decode_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Decode_Chain)
pause
 
[nodeBel,edgeBel,logZ] = UGM_Infer_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Infer_Chain);
nodeBel
pause

samples = UGM_Sample_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Sample_Chain);
for s = 1:10
	figure;
	imagesc(reshape(samples(:,s)',nSeats,nRows)');
end