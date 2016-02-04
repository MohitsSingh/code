function x2(I)
% try the best you can to guess what this is : a 3d array of grayscale
% images, a 4d array of color images, or a cell array.

if length(size(I)) > 2
    if size(I,3) > 3
        II = {};
        for t = 1:size(I,3)
            II{t} = I(:,:,t);
        end
        I = II;
    elseif size(I,4) > 1
        II = {};
        for t = 1:size(I,4)
            II{t} = I(:,:,:,t);
        end
        I = II;
    end
end

if (iscell(I))
    I = mImage(I);
end
figure,imagesc2(I);
end