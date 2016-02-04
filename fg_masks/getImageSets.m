function [trainSet,testSet] = getImageSets(conf)

trainSet = getImageSet(conf.trainSet);

% measure performance.
orig_mode = conf.mode;
conf.mode = 'seg';
testSet = getImageSet(conf.testSet);
conf.mode = orig_mode;

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