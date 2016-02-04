function models = train_model(feats,params,nParts)
%nParts = 3;
models = struct;
[models.models_parts,models.models_links,models.models_shape] = ...
    learn_parts_2(feats.pos_feats,feats.neg_feats,nParts);