function lineseg_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;
config;
conf.get_full_image = true;
%for hand context model
for k = 1:length(indRange)
    i = indRange(k);
    imPath = fullfile(baseDir,d(i).name);
    resPath = fullfile(outDir,strrep(d(i).name,'.jpg','.mat'));
    if (exist(resPath,'file'))
        fprintf(1,'results for image %s already exist.\n', imPath);
        continue;
    end
    I = imread(imPath);
    E = edge(im2double(rgb2gray(I)),'canny');
    [edgelist edgeim] = edgelink(E, []);
    seglist = lineseg(edgelist,3);
    save(resPath,'edgelist','seglist');
end


%LINESEG_ALL Summary of this function goes here
%   Detailed explanation goes here



end

