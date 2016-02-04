function res = global_dnn_feats(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    res.net = init_nn_network();
    load s40_fra;
    res.fra_db = s40_fra;
    res.conf = conf;
    return;
end


net = reqInfo.net;
conf = reqInfo.conf;
conf.get_full_image = true;
I_full = getImage(conf,I);
conf.get_full_image = false;
I_cropped = getImage(conf,I);

[res.full_feat_17,res.full_feat_19] = extractDNNFeats(I_full,net);
[res.crop_feat_17,res.crop_feat_19] = extractDNNFeats(I_cropped,net);

end