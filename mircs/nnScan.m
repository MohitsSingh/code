function [bbs,allScores] = nnScan(conf,I,templates)
[X,uus,vvs,scales,t ] = allFeatures(conf,I,1);
bbs = uv2boxes(conf,uus,vvs,scales,t);
allScores = templates'*X;
end