function [fileNames] = getFileNames(baseDir,suff,ext)
%GETFILENAMES Summary of this function goes here
%   Detailed explanation goes here
d = dir(fullfile(baseDir,['*' suff '*' ext]));
fileNames = {};
for k = 1:length(d)
    fileNames{k} = fullfile(baseDir,d(k).name);
end

end

