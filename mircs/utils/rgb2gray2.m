function I = rgb2gray2(I)
if size(I,3) == 3
    I = rgb2gray(I);
end
end