function [F,D] = colorDesc(im)
%COLORDESC Summary of this function goes here
%   Detailed explanation goes here
    %colorDescriptor <image> --detector <detector> --descriptor <descriptor> --output <descriptorfile.txt';
    
    imPath = getImageFile(globalOpts,train_images{1});
    
    det = 'densesampling --ds_spacing 1';
    desc = 'sift';
    tmp = tempname;
    
   
    cmd = sprintf('%s %s --detector %s --descriptor %s --output %s --outputFormat binary',...
        colorDescPath,imPath,det,desc,tmp);
    
    [status,result] = system(cmd);
    
    [d , f]= readBinaryDescriptors(tmp);
    
    A = dir(tmp);
    A = dlmread(tmp);

end


