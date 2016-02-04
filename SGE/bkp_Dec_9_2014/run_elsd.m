function res = run_elsd(conf,I,reqInfo,moreParams)
if (nargin == 0)
    addpath('~/code/mircs');
    %     cd ~/code/mircs;
    %     initpath;
    %     config;
    %     addpath('/home/amirro/code/3rdparty/elsd_1.0');
    %     res.conf = conf;
    res = [];
    return;
end
I_orig = imread(I);
% [I_orig,I_rect] = getImage(reqInfo.conf,I);
elsdPath = '/home/amirro/code/3rdparty/elsd_1.0';
cd(elsdPath);
[pathstr,name,ext] = fileparts(I);
tmpFile = fullfile(elsdPath,'tmp/', [name '.pgm']);
imwrite(I_orig,tmpFile);
outPath = [tmpFile '.res'];
pgmPath = [tmpFile '.svg'];
cmd = ['./elsd ' tmpFile ' '  outPath];
system(cmd);
A = dlmread(outPath);
[lines ellipses] = parse_svg(A);
res.lineSegments = lines;
res.ellipses = ellipses;
% imagesc(I_orig); axis image; hold on;plot_svg(lines,ellipses);
delete(tmpFile);
delete(outPath);
delete(pgmPath);
end