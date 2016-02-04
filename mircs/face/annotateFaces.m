function face_annotations = annotateFaces(conf,imgData,imgDir,sub,maxPts)
if (nargin < 3 || isempty(imgDir))
    imgDir = conf.imgDir;
end
if (nargin < 5)
    maxPts = 1;
end
annoPath = fullfile('/home/amirro/storage/data/Stanford40/annotations/',sub);
% d = dir(fullfile(imgDir,[str '*.jpg']));

for k = 1:length(imgData)
    [~,name,ext] = fileparts(imgData(k).imageID);
    annoFile = fullfile(annoPath,[name '.mat']);
    if (exist(annoFile,'file'))
        load(annoFile);
        %         clf; imagesc2(getImage(conf,imgData(k)));
        %         hold on; plotPolygons(xy','g+');
        %         drawnow;pause(.1);
        
    else
        %imgPath = getImagePath(conf,imgData(k).imageID);
        %         I = getImage(conf,imgData(k));
        %         imgPath = fullfile(imgDir,imgData{k});
        xy = annotateFacialLandmarks(imgData(k),maxPts);
        save(annoFile,'xy');
    end
    %     face_annotations(k).imgName = imgData(k).im
    face_annotations(k).xy = xy;
end

    function xy = annotateFacialLandmarks(imageData,maxPts)
        I = getImage(conf,imageData);
        % annotate mouths
        clf; imagesc(I); axis image; hold on;
        plotBoxes(imageData.I_rect);
        xy = [];
        n = 0;
        % Loop, picking up the points.
        disp('Left mouse button picks points.')
        disp('Right mouse to quit')
        while n < maxPts
            [xi,yi,but] = ginput(1);
            if (but~=1)
                break;
            end
            plot(xi,yi,'ro')
            n = n+1;
            xy(:,n) = [xi;yi];
        end
        
    end

end

