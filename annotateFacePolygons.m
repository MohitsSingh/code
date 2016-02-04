load fra_db;
alternativeFacesDir = '/home/amirro/storage/data/Stanford40/annotations/faces_alt';
conf.get_full_image = true;
for t = 1:length(fra_db)
    imgData = fra_db(t);
    altPath = j2m(alternativeFacesDir,imgData);
    if exist(altPath,'file'),continue,end
    [objectSamples,objectNames] = getGroundTruth(conf,{imgData.imageID},true);
    if any(cellfun(@any,strfind(objectNames,'face')))
        continue
    end
    
    I = getImage(conf,imgData);
end
