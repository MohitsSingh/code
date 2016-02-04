function sp_agglom_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;config;
addpath(genpath('/home/amirro/code/3rdparty/spagglom_01'));

% addpath('~/code/mircs');
for k = 1:length(indRange)
    currentID=d(indRange(k)).name;
    resPath = j2m(outDir,currentID);
    %imageIndex = findImageIndex(newImageData,currentID);
    conf.get_full_image = false;
    [I,I_rect] = getImage(conf,currentID);
%     regions = get_sp_agglom_regions(M1,'felz');
    loadOrCalc(conf,@get_sp_agglom,I,resPath);
    %fclose(fopen([resPath '.finished'],'a'));
    %fprintf(2,'done with : %s\n:', currentID);
end
fprintf('\n\n ***** finished all files in batch ****\n\n\n\n');

function res = get_sp_agglom(conf,I)
%     sz = size2(I);
%     I = imresize(I,[200 NaN],'bilinear');

% apply at multiple resolutions....

s = size2(I);
res = struct('fg_map','size');
n = 0;
while (s(1) > 48)
    regions = get_sp_agglom_regions(I,'slic');
    n = n+1;
    res(n).fg_map = sum(cat(3,regions{:}),3);
    res(n).size = s;
    I = imResample(I,.8);
    s = size2(I);
end

%     regions = get_sp_agglom_regions(I,'slic');
% res = sum(cat(3,regions{:}),3);
%     res = imresize(res,sz,'bilinear');
