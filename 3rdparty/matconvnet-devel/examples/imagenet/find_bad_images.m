fid = fopen('/home/amirro/bad_imagenet_images.txt');
imgsDir = '/net/mraid11/export/data/amirro/data/ILSVRC2012/images/train/';

line = fgetl(fid)
t=0;
bad_imgs = {};
while (line~=-1)
    t=t+1;
    if strcmp(line(1:4),'CMYK')
%         if (strcmp(line(51:55),'train'))        
            bad_imgs{end+1} = line(51:end);
%         else
%             u = strfind(line,'val');
%             if any(u)
                
%         elseif any(strfind(line,'val'))                   
    end    
    line = fgetl(fid);    
end
fclose(fid);
    
for t = 1:length(bad_imgs)
    bad_imgs{t} = strtrim(bad_imgs{t});
end

% save bad_imgs.mat bad_imgs
% 
% 
% fid = fopen('~/bad_imagenet_images','r');
% line = 0;
% bad_files = {};
% while(line~=-1)
%     line
%     line = fgetl(fid);
%      if (line == -1)
%         break
%      end
%     
%     cmd = sprintf('convert %s -colorspace RGB %s',line,line);
%     [status,result] = system(cmd)
%     
%     %line = strrep(line,'/home/amirro/storage/data/ILSVRC2012_pre/images/','');
% 
%    
% end
% fclose(fid)