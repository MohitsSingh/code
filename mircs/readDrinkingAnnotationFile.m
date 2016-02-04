function m = readDrinkingAnnotationFile(fPath)
fid = fopen(fPath,'r');
%Image Index,Image ID,occlusion/interaction,object type,clear,person yaw,person pitch
[str count] = textscan(fid,'%f %s %f %s %f %f %f %f','Delimiter',',');

x = @(y) mat2cell2(y{1},size(y{1},1));
m = struct('ind',x(str(1)),'imageID',str{2},...
    'occlusion',x(str(3)),'objType',str{4},'yaw',x(str(6)),'pitch',x(str(7)),'obj_orientation',x(str(8)));
fclose(fid);
