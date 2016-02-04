function asker(inputDir,prefix,ext,labelDir,queryString)
%WHATIS Summary of this function goes here
%   Detailed explanation goes here

fileNames = dir(fullfile(inputDir,[prefix '*' ext]));
for k = 1:length(fileNames)
    k
    currentFile = fullfile(inputDir,fileNames(k).name);
    labelFileName = fullfile(labelDir,strrep(fileNames(k).name,ext,'.mat'));
    if (~exist(labelFileName,'file'))
        disp(['no labels exist for ' labelFileName]);
        continue;
    end
    I =  imread(currentFile);
    
    rects = [];
    load(labelFileName);
    
    for iRect = 1:length(rects)
        currentRect = rects(iRect);
        if (~isempty(currentRect.left))
            tl = currentRect.left.tl;
            if (~isempty(tl))
                %                 if (~isfield(currentRect.left,'label'))
                br = currentRect.left.br;
                clf;
                imshow(I);hold on;
                x0 = tl(1); x1 = br(1);
                y0 = tl(2); y1 = br(2);
                plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
                [x,y,z] = ginput(1);
                if (z==1)
                    rects(iRect).left.label = queryString{1};
                elseif (z==2)
                    rects(iRect).left.label = queryString{2};
                end
                %                 end
            end
        end
        if (~isempty(currentRect.right))
            tl = currentRect.right.tl;
            if (~isempty(tl))
                %                 if (~isfield(currentRect.right,'label'))
                br = currentRect.right.br;
                clf;
                imshow(I);hold on;
                x0 = tl(1); x1 = br(1);
                y0 = tl(2); y1 = br(2);
                plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
                [x,y,z] = ginput(1);
                if (z==1)
                    rects(iRect).right.label = queryString{1};
                elseif (z==2)
                    rects(iRect).right.label = queryString{2};
                end
                %                 end
            end
        end
    end
    
    save(labelFileName,'rects');
    
    %     plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
    
    
    %     save(outFileName,'rects');
    
    
end

end




