function do_stuff(nimage)
% for nimage = 3000:3500
%     nimage
% images = [ 3471        3079        3014        3333        3012];

% for nimage = images

save(fullfile('res',sprintf('nimage_%05.0f.mat',nimage)),'nimage');

% nimage
% cd /home/amirro/code/fragments;
% addpath('/home/amirro/data/VOCdevkit/VOCcode/');
% addpath(genpath('/home/amirro/code/3rdparty/SelectiveSearchPcode'));
% 
% VOCinit;
% tic;
% test_images = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s'); %#ok<REMFF1>
% extract_boxes(VOCopts,[],test_images(nimage));
end