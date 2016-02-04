function paths = multiWrite(images,outDir,names)
ensuredir(outDir);
if (nargin < 3)
    names = cell(size(images));
    for k = 1:length(images)
        names{k}=sprintf('%010.0f.jpg',k);
    end
end
for k = 1:length(images)
    k
            
    outFile = fullfile(outDir,names{k});
    paths{k} = outFile;
%     if (~exist(outFile,'file'))
        %images{k} = imresize(images{k},[128 128],'bilinear');
        imwrite(images{k},outFile);
%     end
end
end