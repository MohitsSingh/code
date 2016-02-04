function [trainSet,testSet] = getImageSets(conf)
trainSet = getImageSet(conf.trainSet);
testSet = getImageSet(conf.testSet);

    function imageSet = getImageSet(setName)
        
        if (strcmp(conf.mode,'seg'))
            imageSetPath = sprintf(conf.VOCopts.seg.imgsetpath,setName);
        else
            imageSetPath = sprintf(conf.VOCopts.imgsetpath,setName);
        end
        fid = fopen(imageSetPath);
        trainIDs = textscan(fid,'%s');
        trainIDs = trainIDs{1};
        fclose(fid);
        imageSet = trainIDs;
    end
end