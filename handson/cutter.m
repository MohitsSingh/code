function cut_struct = cutter(inputDir,prefix,ext,labelDir,outDir,max_height)
%WHATIS Summary of this function goes here
%   Detailed explanation goes here
count_ = 0;
if (~exist(outDir,'dir'))
    mkdir(outDir)
end
fileNames = dir(fullfile(inputDir,[prefix '*' ext]));

if (nargin < 6)
    max_height = inf;
end

%cut_struct = struct('labelFileName',{});
for k = 1:length(fileNames)
    k
    currentFile = fullfile(inputDir,fileNames(k).name);
    labelFileName = fullfile(labelDir,strrep(fileNames(k).name,ext,'.mat'));
    
    % create pascal-like annotation from this image.
    rec = PASemptyrecord;
    
    if (~exist(labelFileName,'file'))
        disp(['no labels exist for ' labelFileName]);
        continue;
    end
    I =  imread(currentFile);
    
    rects = [];
    load(labelFileName);
    % not so elegant but I'll live
    nObjs = 0;
    rec.imgname = currentFile;
    rec.imgsize = size(I);
    rec.database = 'WeizmannHands';
    database = rec.database;
    for iRect = 1:length(rects)
        currentRect = rects(iRect);
        if (~isempty(currentRect.left))
            count_ = count_+1;
            nObjs = nObjs+1;
            [bbox,out_path,II] = cutImage(I,currentRect.left,outDir,count_,prefix);
            obj = PASemptyobject;
            %obj.bbox = bbox;
            obj.bbox = [1 1 bbox(3:4)-bbox(1:2)];
            obj.class = prefix;
            obj.view = 'left_hand';
            obj.label = count_;
            imgname = out_path;
            imgsize = size(II);
            objects = obj;
            save(strrep(out_path,'.jpg','_rec.mat'),'imgname','imgsize','database','objects');
        end
        
        if (~isempty(currentRect.right))
            count_ = count_+1;
            nObjs = nObjs+1;
            [bbox,out_path,II] = cutImage(I,currentRect.right,outDir,count_,prefix);
            obj = PASemptyobject;
            %             obj.bbox = bbox;
            obj.bbox = [1 1 bbox(3:4)-bbox(1:2)];
            obj.class = prefix;
            obj.view = 'right_hand';
            obj.label = count_;
            imgname = out_path;
            imgsize = size(II);
            objects = obj;
            save(strrep(out_path,'.jpg','_rec.mat'),'imgname','imgsize','database','objects');
        end
    end
    %#ok<*NASGU>
    
    
    
    
end

% save(fullfile(outDir,'cutstruct.mat'),'cut_struct');


    function [bbox,out_path,II] = cutImage(im,rect,outDir,count_,prefix)
        x0 = rect.tl(1); x1 = rect.br(1);
        y0 = rect.tl(2); y1 = rect.br(2);
        % plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
        
        xmin = min(x0,x1);
        xmax = max(x0,x1);
        ymin = min(y0,y1);
        ymax = max(y0,y1);
        
        % inflate the rect by 50 %
        w = xmax-xmin;
        h = ymax-ymin;
        
        asp_ratio = w/h;
        inflateFactor = 0;%.5;
        xmin = xmin-w*inflateFactor/2;
        ymin = ymin-h*inflateFactor/2;
        xmax = xmax+w*inflateFactor/2;
        ymax = ymax+h*inflateFactor/2;
        
        % but center around xmin, xmax
        
        xmin_ = max(1,xmin);
        ymin_ = max(1,ymin);
        xmax_ = min(size(im,2),xmax);
        ymax_ = min(size(im,1),ymax);
        
        centerx = (xmin+xmax)/2;
        centery = (ymin+ymax)/2;
        
        w = min(centerx-xmin_,xmax_-centerx);
        h = min(centery-ymin_,ymax_-centery);
        
        % finally, keep the aspect ratio,
        if (w / h > asp_ratio)
            w = h * asp_ratio;
        elseif (w / h < asp_ratio)
            h = w / asp_ratio;
        end
        
        ymin = ceil(centery-h);
        ymax = floor(centery+h);
        xmin = ceil(centerx-w);
        xmax = floor(centerx+w);
        II = im(ymin:ymax,xmin:xmax,:);
        
        [s1,s2] = size(II);
        
        resize_factor = 1;
        if (s1 > max_height)
            resize_factor = max_height/s1;
        end
        
        II = imresize(II,resize_factor);
        bbox = [xmin ymin xmax ymax];
        bbox = round(bbox*resize_factor);
        
        out_path = fullfile(outDir,sprintf('%s_%05.0f.jpg',prefix,count_));
        imwrite(II,out_path);
        
    end

end