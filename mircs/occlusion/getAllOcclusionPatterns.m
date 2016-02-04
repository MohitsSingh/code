for k = 1:2:length(newImageData)
    curImageData = newImageData(k);
    k
    resPath = j2m(conf.occludersDir,curImageData);
    if (exist(resPath,'file'))
        continue;
    end
    occlusionPattern = getOcclusionData(conf,curImageData);
    save(resPath,'occlusionPattern');
end