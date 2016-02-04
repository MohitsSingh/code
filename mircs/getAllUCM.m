function ucms = getAllUCM(baseDir,nFiles)
resFile = fullfile(baseDir,'res.mat');
if (exist(resFile,'file'))
    disp('not calculating anything, result already exists.');
    load(resFile);
    return;
end

for k = 1:nFiles
    k
    load(fullfile(baseDir,sprintf('%05.0f.mat',k)));
    ucms{k} = contours2ucm(gPb_orient);
end

save(resFile,'ucms');
end