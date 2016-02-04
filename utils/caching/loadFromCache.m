function data = loadFromCache(conf,obj,pref)
% tries to load an object with parameters given by obj from cache.
% returns empty if cacheDir is undefined or empty.

data = [];
if (nargin < 3)
    pref = [];
else
    pref = [pref '_'];
end
if (isempty(conf.cacheDir))
    return;
else
    hashFileName = [pref num2str(DataHash(obj)) '.mat'];
    cachePath = fullfile(conf.cacheDir,hashFileName);
    % TODO - delete the following if!!
    %     if (~isempty(strfind(pref,'descs')))
    %         dd = dir(fullfile(conf.cacheDir,[pref '*.mat']));
    %         cachePath2 = fullfile(conf.cacheDir, dd.name);
    %         if (~strcmp(cachePath2,cachePath))
    %             movefile(cachePath2,cachePath);
    %         end
    %         return;
    %     end
    if (exist(cachePath,'file'))
        load(cachePath);
    end
end
end