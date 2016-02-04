function graphDatas = constructAllGraphData(conf,imageSet)
gamma_ = conf.shapeGamma;
graphDatas = {};
graphDir =  fullfile(conf.prefix,'data/graph/');
ensuredir(graphDir);
ignoreSave = 0;
for k = 1:length(imageSet)
    k
    currentID = imageSet{k};
    graphPath = fullfile(conf.prefix,'data/graph/',[currentID '_graph.mat']);
    if (~exist(graphPath,'file') || ignoreSave)
        
        [z_bow,z_loc] = getUnaryPotentials(conf,currentID);
        curImage = readImage(conf.VOCopts,imageSet{k});
        z_loc = imfilter(z_loc,fspecial('gauss',conf.gaussian.hsize,conf.gaussian.hsigma));
        prob_image = z_bow.*z_loc.^gamma_;
        prob_image = prob_image / max(prob_image(:));
        superPix_k =  getSuperPix(conf.VOCopts,currentID,'data/superpix',...
            conf.superpixels.coarse_size,conf.superpixels.coarse_regularization);
        superPixFine_k =  getSuperPix(conf.VOCopts,currentID,'data/superpix',...
            conf.superpixels.fine_size,conf.superpixels.fine_regularization);
        r = regionprops(superPix_k,prob_image,'PixelIdxList','MeanIntensity');
        for q = 1:length(r)
            prob_image(r(q).PixelIdxList) = r(q).MeanIntensity;
        end
        graphData = constructGraph(conf.VOCopts,curImage,prob_image,superPixFine_k);
        if (~ignoreSave)
            save(graphPath,'graphData');
        end
    else
        load(graphPath);
    end
    graphDatas{k} = graphData;
end
