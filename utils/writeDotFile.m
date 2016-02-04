function writeDotFile(graphName,G,nodeNames,outFile)
fid = fopen(outFile,'w');

fprintf(fid,'digraph %s {\n',graphName);
if iscell(G)
    graphAsString = G;
else
    graphAsString = cell(size(G));
    
    % convert node names to strings
    if isempty(nodeNames)
        for i = 1:size(G,1)
            for j = 1:2
                graphAsString{i,j} = num2str(G(i,j));
            end
        end
    else
        for i = 1:size(G,1)
            for j = 1:2
                graphAsString{i,j} = nodeNames{G(i,j)};
            end
        end
    end
end
% write edges.
for t = 1:size(G,1)
    fprintf(fid,'\t%s -> %s;\n',graphAsString{t,1},graphAsString{t,2});
end
fprintf(fid,'}\n');
fclose(fid);

