function simple_labeler(inputDir,prefix,ext,outputDir)
%SIMPLE_LABELER Summary of this function goes here
%   Detailed explanation goes here


leftHandString = 'choose left hand (left click to choose, middle to skip)';
rightHandString = 'choose right hand (left click to choose, middle to skip)';


fileNames = dir(fullfile(inputDir,[prefix '*' ext]));
for k = 1:length(fileNames)
    k
    currentFile = fullfile(inputDir,fileNames(k).name);
    outFileName = fullfile(outputDir,strrep(fileNames(k).name,ext,'.mat'));
    if (exist(outFileName,'file'))
        disp(['skipping ' currentFile ', labels already exist']);
%         if (k ~= 148)
            continue;
%         end
    end
    I =  imread(currentFile);
    clf;
    imshow(I);
%     pause;
%     break;
    hold on;
    title(leftHandString);
    rects = struct('left',{},'right',{});
    count_ = 0;
    b = 5;
    while (b ~= 3)
        title(leftHandString);
        [x,y,b] = ginput(1);
        count_ = count_ +1;
        if (b == 1)            
            rects(count_).left.tl = [x y];
            [x,y] = ginput(1);
            rects(count_).left.br = [x y];
            
            tl = rects(count_).left.tl;
            br =  rects(count_).left.br;
            x0 = tl(1); x1 = br(1);
            y0 = tl(2); y1 = br(2);
            
            
            plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
        elseif (b == 3)
            disp('no hands left, loading next image...');
            break;
        else            
            disp('no left hand, switching to right...');            
        end
        
        title(rightHandString);
        [x,y,b] = ginput(1);
        if (b == 1)
            rects(count_).right.tl = [x y];
            [x,y] = ginput(1);
            rects(count_).right.br = [x y];
            
            tl = rects(count_).right.tl;
            br =  rects(count_).right.br;
            x0 = tl(1); x1 = br(1);
            y0 = tl(2); y1 = br(2);
            plot([x0 x1 x1 x0 x0],[y0 y0 y1 y1 y0],'g-','LineWidth',2);
            
        elseif (b == 3)
            disp('no hands left, loading next image...');
            save(outFileName,'rects');
            break;
        else
            disp('no right hand...');
        end
    end
    
    save(outFileName,'rects');
    
    
end

end

