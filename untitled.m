%for t=1:length(res_names)
save_img=true
for t = 1:50:length(all_files)
    [pathstr,name,ext] = fileparts(all_files{t});
              
    if showResOnOrig
        img_name = [imgOrigDir, name, '.jpg'];
    else
        img_name = [img_dir, name, '.ppm'];
    end
    crfBinFile = fullfile(binDir,[name '.bin']);
    img = imread(img_name);
    map = LoadBinFile(crfBinFile, 'int16');
    map(map~=15) = 0;
%     map(1:21,1:10) = repmat((1:21)',1,10);
    if showResOnOrig
        mapRes = map;
        map = zeros(size(img, 1), size(img, 2));
        for cl = unique(mapRes)'
            mapCl = mapRes==cl;
            mapClos = round(double(imresize(mapCl, [size(img, 1) size(img, 2)], 'bilinear')));
            %mapClos = bwperim(mapClos);
            
            map = map + mapClos.*cl;
        end
    end
    
    imgWithMap = img;
    mapRGB     = zeros(size(img));
    cl_txt = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', ...
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', ...
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', ...
        'train', 'tvmonitor'};
    
    fprintf('(%d) classes for %s:', t, name)
    
    clrNum = 0;
    val = 255;
    for cl = unique(map)'
        if cl~=0
            fprintf(' %s;', cl_txt{cl});
            mapCl = map==cl;
            clr = mod(clrNum, 3) + 1;
            if  clrNum<=2
                imClMap = img(:, :, clr);
                imClMap(mapCl) = val;
                imgWithMap(:, :, clr) = imClMap;
                mapRGB(:, :, clr) = val*mapCl;
            else
                imClMap = imgWithMap(:, :, clr);
                imClMap(mapCl) = val;
                imgWithMap(:, :, clr) = imClMap;%(imClMap+imgWithMap(:, :, clr))/2;
                mapRGB(:, :, clr) = (val*mapCl + mapRGB(:, :, clr))/2;                
                clr2 = mod(clr,3)+1;                
                imClMap = imgWithMap(:, :, clr2);
                imClMap(mapCl) = val;                
                imgWithMap(:, :, clr2) = imClMap;%(imClMap+imgWithMap(:, :, clr2))/2;
                mapRGB(:, :, clr2) =  (val*mapCl+mapRGB(:, :, clr2))/2;
                val = (val-1)/2;
            end
            clrNum = clrNum+1;
        end
    end
    fprintf('\n')
    
%     figure(1);
    h = clf;
    mm = 1;
    nn = 3;
    vl_tightsubplot(mm,nn,1); imshow(img); 
    vl_tightsubplot(mm,nn,2); imshow(imgWithMap)
    vl_tightsubplot(mm,nn,3); imshow(mapRGB);
%     subplot(mm,nn,3); imagesc2(map == 15);
    
    
    if save_img
        saveas(h, ['deepLabRes/deepLabRes_', name], 'jpg');
    end
    
%     dpc
    
end