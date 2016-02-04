% try running


cd /home/beny/code/3rd/deepLab/deeplab-deeplab-public-3e413eed0de8
% rmpath('/home/beny/code/3rd/deepLab/deeplab-deeplab-public-3e413eed0de8/matlab/my_script');
addpath('/home/beny/code/3rd/deepLab/deepLabScripts/matlab/');
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
cfrOutputDir = '/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/res_W5_XStd50_RStd3_PosW3_PosXStd3/';
origImagesDir = '~/storage/data/Stanford40/JPEGImages/';
addpath(genpath('~/code/utils'));
all_files = getAllFiles(origImagesDir,'*.jpg');
d = dir('/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/res_W5_XStd50_RStd3_PosW3_PosXStd3/*.bin');

fileNames = {};
for t = 1:length(all_files)
    fileNames{t} = all_files{t}(38:end-4);
end

d = cellfun2(@(x) x(1:end-4) , {d.name});

fileNames = setdiff(fileNames,d);
all_files = fullfile(origImagesDir,cellfun2(@(x) [x '.jpg'],fileNames));

N = length(all_files);
outDirForScript = '../../../../amirro/data/Stanford40/ppm512x512';
batches = batchify(N,1);
cd /home/amirro/code/3rdparty/my_deeplab_scripts/

!setenv LD_LIBRARY_PATH /usr/lib:/usr/local/lib:/usr/lib:/home/beny/matio/installation/lib:/usr/local/cuda-6.5/lib64:/home/beny/anaconda/lib
for iBatch = 1:length(batches)
    fid_list = fopen('/home/amirro/code/3rdparty/my_deeplab_scripts/voc12/list/test_images.txt','w+');
    fid_id = fopen('/home/amirro/code/3rdparty/my_deeplab_scripts/voc12/list/test_images_id.txt','w+');
    curBatch = batches{iBatch};
    files = all_files(curBatch);
    for t = 1:length(files)
        [iBatch t]
        [pathstr,name,ext] = fileparts(files{t});
        fn = [name ext];
        outDir = '/home/amirro/storage/data/Stanford40/ppm512x512';
        origPath = fullfile(origImagesDir,fn);
        I = imread(origPath);
        I = imResample(I,[513 513]);
        out_fn = [name '.ppm'];
        imwrite(I,fullfile(outDir,out_fn));
        pnmPath = fullfile(outDirForScript,out_fn);
        %     fprintf(fid_list,'%s\n',pnmPath);
        fprintf(fid_list,'%s\n',out_fn);
        fprintf(fid_id,'%s\n',name);
    end
    fclose(fid_list);
    fclose(fid_id);
    
    % run the fc8    
    !setenv LD_LIBRARY_PATH /usr/lib:/usr/local/lib:/usr/lib:/home/beny/matio/installation/lib:/usr/local/cuda-6.5/lib64:/home/beny/anaconda/lib
    
    
    !./run_deepLab_vqa.sh
    % run the crf
    !./run_densecrf_vqa.sh
    
    !./run_densecrf_vqa_tmp.sh
    % delete the fc8
    for t = 1:length(files)
        !rm /home/amirro/storage/data/Stanford40/deeplab_workdir/test_images/fc8/*.mat
    end    
end

%% read the binaries...
% 
% %binDir = '/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/test_images/fc8';
% for t = 1:
%     [pathstr,name,ext] = fileparts(files{t});
%     %bin_fn = fullfile(binDir,[name '_blob_0.bin']);
%     bin_fn = fullfile(binDir,[name '.bin']);
%     a = LoadBinFile(bin_fn,'int16');
% end


%A = load('/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/test_images/fc8/applauding_001_blob_0.mat');

%% show results...

%


%binDir      = '/home/beny/code/3rd/deepLab/deepLabScripts/tmp/res/features/net/test_images/fc8/post_densecrf_W10_XStd100_RStd10_PosW3_PosXStd3_numSample100/'; % one possible res dir
img_dir       = '/home/amirro/storage/data/Stanford40/ppm512x512/';
imgOrigDir    = origImagesDir;
showResOnOrig = 1;
save_img      = 0;

%res_names = get_filenames(binDir, 'bin');
% res_names =

%for t=1:length(res_names)
for t = 1:length(all_files)
    [pathstr,name,ext] = fileparts(all_files{t});
    if showResOnOrig
        img_name = [imgOrigDir, name, '.jpg'];
    else
        img_name = [img_dir, name, '.ppm'];
    end
    crfBinFile = fullfile(cfrOutputDir,[name '.bin']);
    img = imread(img_name);
    map = LoadBinFile(crfBinFile, 'int16');
    
    if showResOnOrig
        mapRes = map;
        map = zeros(size(img, 1), size(img, 2));
        for cl = unique(mapRes)'
            mapCl = mapRes==cl;
            mapClos = round(double(imresize(mapCl, [size(img, 1) size(img, 2)], 'bilinear')));
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
    
    h = figure(1)
    subplot(2,2,1); imshow(imgWithMap);
    subplot(2,2,2); imshow(mapRGB);
    subplot(2,2,3); imagesc2(map == 15);
    
    if save_img
        saveas(h, ['deepLabRes_', name], 'jpg');
    end
    
    dpc
    
end

!./run_densecrf_vqa_tmp.sh
name = '1'
cfrOutputDir='/home/amirro/code/3rdparty/matconvnet-fcn-master/res_W5_XStd50_RStd3_PosW3_PosXStd3'
crfBinFile = fullfile(cfrOutputDir,[name '.bin']);
map = LoadBinFile(crfBinFile, 'int16');
figure,imagesc(map)
figure,imagesc(I);



