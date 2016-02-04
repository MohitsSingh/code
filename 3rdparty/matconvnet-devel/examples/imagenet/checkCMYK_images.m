imgsDir = '/net/mraid11/export/data/amirro/data/ILSVRC2012/images/train/';
fid = fopen('/home/amirro/x.txt')
c = makecform('cmyk2srgb'); % Only need to be called once
for t = 1:22
    t
    line = fgetl(fid);
    subdir = line(1:9);
    imgPath = fullfile(imgsDir,subdir,line);
    imt = vl_imreadjpeg({imgPath});
    imt = imt{1};
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    elseif size(imt,3) == 4 % CMYK image
        imt(isnan(imt)) = 0;
        imt = single(applycform(double(imt), c) .* 255); % Turn CMYK into RGB image
    end
    clf; imagesc(imt/255);
%     pause;
end
fclose(fid);




