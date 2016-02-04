function boxes_parallel(baseDir,d,indRange,outDir)

addpath(genpath(fullfile('~/code/3rdparty/SelectiveSearchPcode')));
addpath('/home/amirro/code/fragments/boxes');

for k = 1:length(indRange)
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    if (exist(resFileName,'file'))
        continue;
    end
    I = imread(imagePath);
    boxes = SelectiveSearchBoxes(I);
    boxes = uint16(boxes(:,[2 1 4 3])); % xmin ymin xmax ymax
    save(resFileName,'boxes');
    fprintf('done with image %s!\n',filename);
end

fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end