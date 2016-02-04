function [z_bows,z_locs] = constructUnaryPotentials(conf,imageSet,IB,IG)
n_bow_neighbors = conf.n_bow_neighbors;
n_gist_neighbors = conf.n_gist_neighbors;
unaryPath = fullfile(conf.prefix,'unary.mat');
nnn =length(imageSet);

% pre-load structs...
if (strcmp(conf.mode,bb))
    recsPath = fullfile(conf.prefix,'recs.mat');
    if (exist(recsPath,'file'))
        load(recsPath);
    else
        recs = readAllRecs(conf.conf.VOCopts,imageSet);
        save(recsPath,'recs');
    end
end

if (~exist(unaryPath,'file'))
    z_bows = {};
    z_locs = {};
    for k = 1:nnn
        imageId = imageSet{k};
        [z_bow,z_loc] = getUnaryPotentials(conf,imageId,IG,IB);
        z_bows{k} = z_bow;
        z_locs{k} = z_loc;
        %figure(3),imagesc(z_bow);pause;
    end
    save(unaryPath,'z_bows','z_locs');
else
    load(unaryPath);
end

end