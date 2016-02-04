function [z_bows,z_locs] = constructAllUnaryPotentials(conf,trainSet,testSet,IB,IG,dict)
unaryPath = fullfile(conf.prefix,'unary.mat');
nnn =length(testSet);

% pre-load structs...
if (strcmp(conf.mode,'bb'))
    recsPath = fullfile(conf.prefix,'recs.mat');
    if (exist(recsPath,'file'))
        load(recsPath);
    else
        recs = readAllRecs(conf.VOCopts,trainSet);
        save(recsPath,'recs');
    end
end
unaryFeatDir = fullfile(conf.prefix,'data/unary');

if (conf.preLoadFeatures) % faster but more memory consuming
    [bowFeatures,bowFrames] = getBows(conf.VOCopts,trainSet,dict,'data/bow');
end

ensuredir(unaryFeatDir);
if (~exist(unaryPath,'file'))
    z_bows = {};
    z_locs = {};
    for k = 1:nnn
        k
        imageId = testSet{k};
        if (conf.preLoadFeatures)
            [z_bow,z_loc] = getUnaryPotentials(conf,imageId,trainSet,IG(:,k),IB(:,k),dict,...
                bowFeatures,bowFrames);
        else
            [z_bow,z_loc] = getUnaryPotentials(conf,imageId,trainSet,IG(:,k),IB(:,k),dict);
        end
        if (nargout > 0)
            z_bows{k} = z_bow;
            if (nargout > 1)
                z_locs{k} = z_loc;
            end
        end
        %figure(3),imagesc(z_bow);pause;
    end
    save(unaryPath,'z_bows','z_locs');
else
    load(unaryPath);
end

end