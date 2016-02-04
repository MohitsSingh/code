function res = extract_dnn_feats_masks(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    addpath('~/code/mircs');
    initpath;
    config;
    vl_setup;
    vl_setupnn;
    addpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained');
    install
    res.featureExtractor = DeepFeatureExtractor(conf);
    return;
end

mcgDir = '~/storage/fra_face_seg';
L = load(j2m(mcgDir,params.name));
cands = L.res.candidates;

masks = cands2masks(cands.cand_labels, cands.f_lp, cands.f_ms);
regions = {};
for t = 1:size(masks,3)
    regions{t} = masks(:,:,t);
end

res.feats = initData.featureExtractor.extractFeatures(params.img,regions);

