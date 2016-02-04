function imageData = detect_occluders(conf,imageData)
    imageIDs = {imageData.imageID};
    for k = 1:length(imageData)
        k
        currentName = strrep(imageIDs{k},'.jpg','.mat');
        imageData(k).occluders = load(fullfile(conf.occludersDir,currentName));        
    end
end