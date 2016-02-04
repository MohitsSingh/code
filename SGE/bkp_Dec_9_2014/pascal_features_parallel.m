function res = pascal_features_parallel(conf,I,reqInfo)
if (nargin == 0)
    cd /home/amirro/code/ruining_context;
    default_init;
    res.imdb = imdb;
    res.encoder = encoder;
    res.model = model;
    res.opts = opts;
    res.class_names = imdb.meta.classes;
    return;
end
[pathstr,name,ext] = fileparts(I);
I = im2double(imread(I));
imgIndex = find(cellfun(@any,strfind(reqInfo.imdb.images.name,[name ext])));
objects = reqInfo.imdb.objects{imgIndex};
% now, "ruin" the context for a specific class - e.g, horse
cls = 'horse';
res.desc_orig = my_encode_image(reqInfo.encoder,I,reqInfo.opts);

I_noise = ruin_context(I,objects,cls,1);
res.desc_ruined_noise = my_encode_image(reqInfo.encoder,I_noise,reqInfo.opts);
I_blur = ruin_context(I,objects,cls,2);
res.desc_ruined_blur = my_encode_image(reqInfo.encoder,I_blur,reqInfo.opts);
I_no_context = ruin_context(I,objects,cls,3);
res.desc_no_context = my_encode_image(reqInfo.encoder,I_no_context,reqInfo.opts);
I_no_object = ruin_context(I,objects,cls,4);
res.desc_ruined_no_object = my_encode_image(reqInfo.encoder,I_no_object,reqInfo.opts);