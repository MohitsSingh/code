function [paths,names] = getAllFiles(baseDir,ext)

if (nargin < 2)
    ext = '';
end
if (~iscell(ext))
    ext = {ext};
end

paths = {};
names = {};
for k = 1:length(ext)
    d = dir(fullfile(baseDir,['*' ext{k}]));
    for kk = 1:length(d)
        curName = d(kk).name;
        names{end+1} = curName;
        paths{end+1} = fullfile(baseDir,curName);
    end
end

end