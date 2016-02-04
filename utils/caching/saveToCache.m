function saveToCache(conf,obj,data,pref) %#ok<INUSL,*INUSD>
%SAVETOCACHE Saves data to cache using obj as a description for the data.
%   Detailed explanation goes here
if (isempty(conf.cacheDir))
    return;
end

if (nargin < 4)
    pref = [];
else
    pref = [pref '_'];
end


hashFileName = [pref num2str(DataHash(obj)) '.mat'];
cachePath = fullfile(conf.cacheDir,hashFileName);
if (exist(cachePath,'file'))
    warning('cache file already exists - not saving');
else
    save(cachePath,'data');
end

end