
clear, clc,
close all
addpath('Funcs');

%% 1. Parameter Settings
doFrameRemoving = true;
useSP = true;           %You can set useSP = false to use regular grid for speed consideration
doMAEEval = true;       %Evaluate MAE measure after saliency map calculation
doPRCEval = true;       %Evaluate PR Curves after saliency map calculation

SRC = '~/storage/data/Stanford40/JPEGImages';
BDCON = 'Data/BDCON';   %Path for saving bdCon feature image
SP = 'Data/SP';         %Path for saving superpixel index image and mean color image
RES = 'Data/s40_sal_res_fine';       %Path for saving saliency maps
srcSuffix = '.jpg';     %suffix for your input image

if ~exist(SP, 'dir')
    mkdir(SP);
end
if ~exist(BDCON, 'dir')
    mkdir(BDCON);
end
if ~exist(RES, 'dir')
    mkdir(RES);
end
%% 2. Saliency Map Calculation

% wantedClasses = {'drinking','smoking','blowing_bubbles','brushing_teeth'};
% files = [];
% for iClass = 1:length(wantedClasses)
    files = dir(fullfile(SRC,'drin*.jpg'));
% end

%% 
opts.show = false;
for k=1:1:length(files)                
    srcName = files(k).name;
    noSuffixName = srcName(1:end-length(srcSuffix));    
    resPath = fullfile(RES,[noSuffixName '.mat']);
    if (exist(resPath,'file'))
        continue;
    end
    disp(k);
    maxImageSize = 100;   
    opts.maxImageSize = maxImageSize;    
    spSize = 10;
    filePath = fullfile(SRC, srcName);
    srcImg = imread(filePath);
    opts.pixNumInSP = spSize;
    [res,res_bd] = extractSaliencyMap(srcImg,opts);
    clf;imagesc(res);axis image;drawnow;pause;
    
    
    
%     save(resPath,'srcName','res','res_bd','opts');
end

