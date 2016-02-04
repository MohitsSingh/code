function [rects2] = selectSamples2(conf,ids,saveDir)
%SELECTSAMPLES selects rectangles of interest from images,
%   possibly saving each selected rectangle to a specified directory.
if (nargin == 3)
    mkdir(saveDir);
end
% figure(1);
for k = 1:length(ids)
    fName = fullfile(saveDir, [num2str(k) '.mat']);
    if (mod(k,100)==0)
        k
    end
    
    if (nargin == 3 && exist(fName,'file'))
        load(fName);
        rects2{k} = bw;
        continue;
    else
        
        currentID = ids{k};
        I = getImage(conf,currentID);
        close all;
        % %         figure(1);
        %         cla;
        imshow(I);
        bw = roipoly(I);
        rects2{k} = bw;
        if (nargin == 3)
            save(fName,'bw');
        end
    end
end
close all;
end

