function landmarks = detect_landmarks(conf,images,toScale,useTopHalf)

landmarks = cell(1,length(images));

% load face_p146_small.mat
% load face_p99.mat;
load multipie_independent.mat;


% 5 levels for each octave
model.interval = 5;
% set up the threshold
% model.thresh = min(-.9, model.thresh);
model.thresh = min(-1.5, model.thresh);

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

for iImage = 1:length(images),
    fprintf('testing: %d/%d\n', iImage, length(images));
    %     im = imread(fullfile('images',images(iImage).name));
    im = getImage(conf,images{iImage});
    %im = images{iImage};
    if (nargin >= 3)
    
        im = imresize(im,toScale,'bilinear');
    end
    
    if (nargin == 4 && useTopHalf)
        im = im(1:floor(end/2),:,:);
    end
    
    %     clf; imagesc(im); axis image; axis off; drawnow;
    
    tic;
    bs = detect(im, model, model.thresh);
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);
    im = im2double(im);
    im = min(im,1);
    im = max(im,0);
    if (isempty(bs))
        disp(['warning - no faces detected for image ' num2str(iImage)]);
        %         pause;
        continue;
    end
    
    dettime = toc;
    
    % show highest scoring one
%         figure,showboxes(im, bs(1),posemap),title('Highest scoring detection');
%         pause;
%     show all
%         figure,showboxes(im, bs,posemap),title('All detections above the threshold');
%     
    fprintf('Detection took %.1f seconds\n',dettime);
    landmarks{iImage} = bs;
    %     disp('press any key to continue');
    %     pause;
    %     close all;
end
disp('done!');

end