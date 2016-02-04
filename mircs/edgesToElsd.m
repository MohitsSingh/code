function A = edgesToElsd(edgeMap)
curDir = pwd;
elsdDir = '/home/amirro/code/3rdparty/elsd_1.0/';
cmd = './elsd 3.pgm 3.res'
r = double(edgeMap > 0);
rprops = regionprops(bwlabel(r),'Area','PixelIdxList');
for k = 1:length(rprops)
    if (rprops(k).Area < 5)
        %             k
        r(rprops(k).PixelIdxList) = 0;
    end
end

%     clf;imshow(imfilter(r,fspecial('gauss',19,2)));

%     r = imfilter(r,fspecial('gauss',5,.5));
%     r = imrotate(r,5,'bilinear','crop');
%     imwrite(r,'/home/amirro/code/3rdparty/elsd_1.0/3.pgm');
imwrite(imresize(r,1,'bilinear'),'/home/amirro/code/3rdparty/elsd_1.0/3.pgm');
cd(elsdDir);
[status,result] = system(cmd);
d = dir('/home/amirro/code/3rdparty/elsd_1.0/3.res');
A = [];
if (d.bytes > 0)
    A = dlmread('/home/amirro/code/3rdparty/elsd_1.0/3.res');
end
cd(curDir);