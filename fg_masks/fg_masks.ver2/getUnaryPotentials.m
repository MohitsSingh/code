function [z_bow,z_loc] = getUnaryPotentials(conf,imageID,trainSet,IG,IB,dict,...
    allFeats,allFrames)
curFeatPath = fullfile(conf.prefix,'data/unary',[imageID '_feat.mat']);
if (~exist(curFeatPath,'file'))
    bow_neighbors = IB(1:conf.n_bow_neighbors);
    %             curImage = readImage(conf.VOCopts,imageID );
    [bowFeatures,bowFrames] = getBows(conf.VOCopts,{imageID},dict,'data/bow');
    fr = bowFrames{1};
    feat = bowFeatures{1}(:);
    if (nargin <= 6)
        [bowFeatures,bowFrames] = getBows(conf.VOCopts,trainSet(bow_neighbors),dict,'data/bow');
    else
        bowFeatures = allFeats(bow_neighbors);
        bowFrames = allFrames(bow_neighbors);
    end
    superPix_k =  getSuperPix(conf.VOCopts,imageID,'data/superpix',...
        conf.superpixels.coarse_size,conf.superpixels.coarse_regularization);
    z_loc = zeros(size(superPix_k));
    gist_neighbors = IG(1:conf.n_gist_neighbors);
    
    if (strcmp(conf.mode,'bb'))
        Pw = unaryStats(conf.VOCopts,trainSet(bow_neighbors),...
            bowFeatures(bow_neighbors),bowFrames(bow_neighbors),dict,recs(bow_neighbors));
        for t = 1:length(gist_neighbors)
            fg_mask = pasRec2Mask(conf.VOCopts,trainSet{gist_neighbors(t)},...
                recs{gist_neighbors(t)},size(z_loc));
            z_loc = z_loc + imresize(fg_mask,size(z_loc),'nearest');
        end
    else
        Pw = unaryStats_bb(conf.VOCopts,trainSet(bow_neighbors),...
            bowFeatures,bowFrames,dict);
        for t = 1:length(gist_neighbors)
            fg_mask = imread(sprintf(conf.VOCopts.seg.clsimgpath,trainSet{gist_neighbors(t)}));
            fg_mask = fg_mask > 0;
            z_loc = z_loc + imresize(fg_mask,size(z_loc),'nearest');
        end
    end
    
    z_loc = z_loc/length(gist_neighbors);
    z_bow = paintStats(conf.VOCopts,imageID,Pw, fr,feat,superPix_k,1);
    
    %experimental- recalculate according new new weight....
    % %         Pw2 = unaryStats(conf.VOCopts,{imageID},...
    % %             bowFeatures(k),bowFrames(k),dict,{z_bow});%{z_bow>mean(z_bow(:))});
    % %         % recalculate according new new weight....
    % %         z_bow = paintStats(conf.VOCopts,imageID,Pw2, fr,feat,superPix_k,1);
    
    z_bow  = z_bow/max(z_bow(:));
    save(curFeatPath,'z_bow','z_loc');
else
    load(curFeatPath);
end