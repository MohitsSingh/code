function [ paths,bbs ] = load_coco_bbs( baseDir ,inputPath)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% function [ bb_data ] = loadBBS( baseDir, inputPath )
%LOADBBS Summary of this function goes here
%   Detailed explanation goes here

paths = {};
bbs = {};
fid = fopen(inputPath);
s = fgetl(fid);
t = 0;
while s~=-1
    %         if (t >=1000),break,end
    t = t+1;
    if (mod(t,500)==0)
        disp(t)
    end
    paths{end+1} = s;
    f = fgetl(fid);
    bbs{end+1} = f;
    s = fgetl(fid);
    
    %         I = imread(fullfile(baseDir,paths{end}));
    %         curBB = str2num(bbs{end});
    %         curBB(3:4) = curBB(3:4)+curBB(1:2);
    %         clf; imagesc2(I); plotBoxes(curBB);
    %         pause;
end
fclose(fid)

bbs = cellfun2(@str2num,bbs);
bbs = cat(1,bbs{:});
bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
% [b,ib] = sort(paths);
% for iu = 1:length(paths)
%     u = ib(iu);
%     I = imread(fullfile(baseDir,paths{u}));
%     curBB = bbs(u,:);
%     curBB(3:4) = curBB(3:4)+curBB(1:2);
%     clf; imagesc2(I); plotBoxes(curBB);
%     drawnow
%     pause
% end



% end



end

