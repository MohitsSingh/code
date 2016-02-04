function res = imageStackToCell(IJ)
res = {};
if (length(size(IJ==3)))
    for t = 1:size(IJ,3)
        res{t} = squeeze(IJ(:,:,t));
    end
else
    for t = 1:size(IJ,4)
        res{t} = squeeze(IJ(:,:,:,t));
    end
end
end