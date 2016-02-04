function [z_bows,z_locs] = constructAllUnaryPotentials(conf,trainSet,testSet,IB,IG,dict)
unaryPath = fullfile(conf.prefix,'unary.mat');
nnn =length(testSet);
recs = [];
debug_mode = false;
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
if (~exist(unaryPath,'file') || debug_mode);
    z_bows = {};
    z_locs = {};
    
    if (debug_mode)
        load iu;
    end
%     iu = 413;
    for ik = 1:nnn
        if (debug_mode)
            k = iu(ik)
             curImage = readImage(conf.VOCopts,testSet{k});
             imshow(curImage);pause;
        else
            k = ik
        end
        imageId = testSet{k};
        if (conf.preLoadFeatures)
            [z_bow,z_loc] = getUnaryPotentials(conf,imageId,trainSet,IG(:,k),IB(:,k),dict,...
                bowFeatures,bowFrames,recs);
        else
            [z_bow,z_loc] = getUnaryPotentials(conf,imageId,trainSet,IG(:,k),IB(:,k),dict);
        end
        
        if (debug_mode)
            m_loc = {};
            m_bow = {};
            for q = 1:conf.n_bow_neighbors
                m_bow{q} =  readImage(conf.VOCopts,trainSet{IB(q,k)});
            end
            for q = 1:conf.n_gist_neighbors
                m_loc{q} =  readImage(conf.VOCopts,trainSet{IG(q,k)});
            end
        end
        
        if (nargout > 0)
            z_bows{k} = z_bow;
            if (nargout > 1)
                z_locs{k} = z_loc;
            end
        end
        %figure(3),imagesc(z_bow);pause;
    end
    if (~debug_mode)
        save(unaryPath,'z_bows','z_locs');
    end
else
    load(unaryPath);
end

end