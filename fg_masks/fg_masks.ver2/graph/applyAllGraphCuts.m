function res = applyAllGraphCuts(conf,imageSet,graphDatas,ignoreSave)

res = {};
resPath =  fullfile(conf.prefix,'data/res_all.mat');
debug_ = false;
if (nargin < 4)
    ignoreSave = 0;
end
if (~exist(resPath,'file') || ignoreSave)
    for ik = 1:length(imageSet)
        k = ik
%         k = 5
        superPixMap =  getSuperPix(conf.VOCopts,imageSet{k},'data/superpix',...
            conf.superpixels.fine_size,conf.superpixels.fine_regularization);
        curImage = readImage(conf.VOCopts,imageSet{k});
        res{k} =  applyGraphcut(curImage,superPixMap,graphDatas{k}) > 0;        
        if (debug_)
            curImage = readImage(conf.VOCopts,imageSet{k});
            %[z_bow,z_loc] = getUnaryPotentials(conf,imageSet{k})
            %z_loc = z_locs{k};
            %z_loc = imfilter(z_loc,fspecial('gauss',19,7));
            %prob_image = z_bows{k}.*z_loc.^gamma_;
            qqq = [curImage;im2uint8(jettify(res{k}))];
            figure(1);imshow(qqq);
            pause;
        end
    end
    if (~ignoreSave)
        save(resPath,'res');
    end
else
    load(resPath);
end