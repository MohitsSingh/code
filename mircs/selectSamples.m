function [rects] = selectSamples(conf,ids,saveDir)
%SELECTSAMPLES selects rectangles of interest from images,
%   possibly saving each selected rectangle to a specified directory.
if (nargin == 3)
    if (~exist(saveDir,'dir'))
        mkdir(saveDir);
    end
end
% figure(1);
for k = 1:length(ids)
            
    fName = fullfile(saveDir, [num2str(k) '.mat']);
%     if (mod(k,100)==0)
%         k
%     end
    
    if (nargin == 3 && exist(fName,'file'))
        load(fName);
        rects{k} = position;
        
%         [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
%          hold on;
%                plotBoxes2([ymin xmin ymax xmax]);
        continue;
    else
        fprintf('%d to go out of %d\n',length(ids)-k,length(ids));
        currentID = ids{k};
        [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
        close all;
        % %         figure(1);
        %         cla;
        imshow(I);
        if (xmax > 0)
               hold on;
               plotBoxes2([ymin xmin ymax xmax]);
        end
        h = imrect;
        position = wait(h);
        rects{k} = position;
        if (nargin == 3)
            save(fName,'position');
        end
    end
end
close all;
end

