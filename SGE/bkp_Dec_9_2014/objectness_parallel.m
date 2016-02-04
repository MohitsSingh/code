function objectness_parallel(baseDir,d,indRange,outDir)

cd /home/amirro/code/3rdparty/objectness-release-v2.0;
startup;
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    %     fprintf('checking if results for image %s exist...',filename);
    if (exist(resFileName,'file'))
        fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating...');
    end
    imgExample = imread(imagePath);
    boxes = runObjectness(imgExample,1000);
    boxes = single(boxes);
    save(resFileName,'boxes');
end
fprintf('\n\n\nFINISHED\n\n\n!\n');
end