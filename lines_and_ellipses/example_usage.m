% before running this script, you should run the script: elsd_1.0/batch_apply.sh.
% the shell script takes files of the name : <basedir>/filename.jpg
% and transforms them to files named: <outdir>/filename.pgm.svg.
% remember to modify the paths inside the script!

% the .svg files can be visualized by this (example_usage.m) script.

inputDir = '...';
outputDir = '...';

imageName = 'example.jpg';

imPath = fullfile(inputDir,imageName);
elsdResultPath = fullfile(outputDir,strrep(imageName,'.jpg','.pgm.svg'));

% read contents of file
A = dlmread(elsdResultPath);
% parse to lines and ellipses
[lines_,ellipses_] = parse_svg(A,I_rect(1:2));

% visualize results
I = imread(elsdResultPath);
clf; imagesc(I); axis image; hold on;

for iLine = 1:size(lines_,1)
    a = lines_(iLine,:);
    plot(a([1 3]),a([2 4]),'r','LineWidth',2);
end

for iEllipse = 1:size(ellipses_,1)
    a = ellipses_(iEllipse,:);
    % x,y, are plotted coordinates of ellipses. you don't strictly need
    % them.
    [h,x,y] = plotEllipse2(a(1),a(2),a(3),a(4),a(5:7),'g',100,2,[],true);
end