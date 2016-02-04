function I = myResize(I,height)
height_ratio = height/size(I,1);
if (height_ratio <= 1 && isa(I,'double'))
    I = resize(I,height_ratio);
else
    I = imresize(I,height_ratio,'bicubic');
end
