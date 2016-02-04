function t = get_pyramid(I,conf)
params = conf.detection.params;

[t.hog, t.scales] = esvm_pyramid(I, params);

t.padder = params.detect_pyramid_padding;
for level = 1:length(t.hog)
    t_level = t.hog{level};
    t_level = padarray(t_level, [t.padder t.padder 0], 0);

    t.hog{level} = t_level;
end

minsizes = cellfun(@(x)min([size(x,1) size(x,2)]), t.hog);
t.hog = t.hog(minsizes >= t.padder*2);
t.scales = t.scales(minsizes >= t.padder*2);
end