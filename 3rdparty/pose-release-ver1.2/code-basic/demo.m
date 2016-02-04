compile;

% load and display model
load('BUFFY_final');
visualizemodel(model);
disp('model template visualization');
disp('press any key to continue'); 
pause;
visualizeskeleton(model);
disp('model tree visualization');
disp('press any key to continue'); 
pause;
baseDir = '~/storage/data/Stanford40/JPEGImages/';
imlist = dir(fullfile(baseDir,'smok*.jpg'));
%%
for i = 1:length(imlist)
    % load and display image
    im = imread(fullfile(baseDir,imlist(i).name));
    
%     im = imread('~/storage/data/Stanford40/JPEGImages/drinking_001.jpg');
    clf; imagesc(im); axis image; axis off; drawnow;
    im = imresize(im,[200 NaN]);
    % call detect function
    tic;
    boxes = detect(im, model, min(model.thresh,-1));
    dettime = toc; % record cpu time
    boxes = nms(boxes, .1); % nonmaximal suppression
    colorset = {'g','g','y','m','m','m','m','y','y','y','r','r','r','r','y','c','c','c','c','y','y','y','b','b','b','b'};
    showboxes(im, boxes(1,:),colorset); % show the best detection
%     showboxes(im, boxes,colorset);  % show all detections
    fprintf('detection took %.1f seconds\n',dettime);
    disp('press any key to continue');
    pause;
end

disp('done');
