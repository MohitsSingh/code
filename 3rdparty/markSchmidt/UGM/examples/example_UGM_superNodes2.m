%% Make noisy X
getNoisyX

%% Make Blocks

nodeNums = reshape(1:nNodes,nRows,nCols);
superNodes = cell(0,1);
for j = 1:2:nCols-1
    for i = 1:2:nRows-1
        superNodes{end+1,1} = [nodeNums(i,j) nodeNums(i+1,j) nodeNums(i,j+1) nodeNums(i+1,j+1)];
    end
end

%% ICM on super nodes

% Regular ICM
fprintf('Decoding with ICM...\n');
ICMDecoding = UGM_Decode_ICM(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(ICMDecoding,nRows,nCols));
colormap gray
title('ICM Decoding of Noisy X');
fprintf('(paused)\n');
pause

% ICM on super nodes
fprintf('Decoding with ICM on super nodes...\n');
superICMdecoding = UGM_Decode_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Decode_ICM);

figure;
imagesc(reshape(superICMdecoding,nRows,nCols));
colormap gray
title('Block ICM Decoding of Noisy X');
fprintf('(paused)\n');
pause

%% Generalized Belief Propagation

% Regular Mean Field
fprintf('Running Loopy Belief Propagtion Inference...\n');
[nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);

figure;
imagesc(reshape(nodeBelLBP(:,2),nRows,nCols));
colormap gray
title('Loopy Belief Propagation Estimates of Marginals');
fprintf('(paused)\n');
pause

% Generalized Belief Propagation
fprintf('Running Generalized Belief Propagation Inference...\n');
[nodeBelGBP,edgeBelGBP,logZGBP] = UGM_Infer_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Infer_LBP);

figure;
imagesc(reshape(nodeBelGBP(:,2),nRows,nCols));
colormap gray
title('Generalized Belief Propagation Estimates of Marginals');
fprintf('(paused)\n');
pause

%% Gibbs Sampling on Super Nodes

% Regular Gibbs Sampling
fprintf('Running Gibbs Sampler...\n');
burnIn = 10;
edgeStruct.maxIter = 20;
samplesGibbs = UGM_Sample_Gibbs(nodePot,edgePot,edgeStruct,burnIn);

figure;
for i = 1:10
    subplot(2,5,i);
    imagesc(reshape(samplesGibbs(:,i*edgeStruct.maxIter/10),nRows,nCols));
    colormap gray
end
suptitle('Samples from Gibbs sampler');

figure;
imagesc(reshape(mean(samplesGibbs,2),nRows,nCols));
colormap gray
title('Gibbs Estimates of Marginals');
fprintf('(paused)\n');
pause

% Gibbs Sampling on Super Nodes
fprintf('Running Gibbs Sampler on Super Nodes...\n');
samplesBlockGibbs = UGM_Sample_SuperNode(nodePot,edgePot,edgeStruct,superNodes,@UGM_Sample_Gibbs,burnIn);

figure;
for i = 1:10
    subplot(2,5,i);
    imagesc(reshape(samplesBlockGibbs(:,i*edgeStruct.maxIter/10),nRows,nCols));
    colormap gray
end
suptitle('Samples from Block Gibbs sampler');

figure;
imagesc(reshape(mean(samplesBlockGibbs,2),nRows,nCols));
colormap gray
title('Gibbs Estimates of Marginals');
fprintf('(paused)\n');
pause