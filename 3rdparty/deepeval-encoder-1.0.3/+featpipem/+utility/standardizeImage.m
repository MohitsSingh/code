function im = standardizeImage(im)
%STANDARDIZEIMAGE Prepare image for encoding using Caffe

    im = im2single(im);

    if ndims(im) == 2
        %error('Dimension expansion disabled');
        im = repmat(im, [1 1 3]);
    end

    if ndims(im) ~= 3
        error('Image must be a three channel RGB image');
    end

end
