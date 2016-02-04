function G = GlobalDescriptor(I)

if (length(size(I)) == 2)
    I = cat(3,I,I,I); % artificial coloring
end


m = 240*320/numel(I);
if (m < 1)
    I = imresize(I,m,'bilinear');
end
imageSize = size(I);


orientationsPerScale = [8 8 4];
numberBlocks = 4;

% Precompute filter transfer functions (only need to do this one, unless image size is changes):
G = createGabor2(orientationsPerScale, imageSize(1), imageSize(2));

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
output = prefilt(double(I), 4);
G = gistGabor(output, numberBlocks, G);

end
