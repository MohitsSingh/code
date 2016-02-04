function globalFeats = getGlobalFeatures(conf,fra_db,featureExtractor)
globalFeats = struct('global',{},'person',{});
I_full = {};
I_person = {};
I_face = {};
tic_id = ticStatus('extracting global features...',.5,.5);
for iImg = 1:length(fra_db)
    [I,I_rect] = getImage(conf,fra_db(iImg));
    I_full{iImg} = I;
    I_person{iImg} = cropper(I,I_rect);
    I_face{iImg} = cropper(I,round(fra_db(iImg).faceBox));
    tocStatus(tic_id,iImg/length(fra_db));
end

globalFeats(1).global = featureExtractor.extractFeaturesMulti(I_full);
globalFeats(1).person = featureExtractor.extractFeaturesMulti(I_person);
globalFeats(1).face = featureExtractor.extractFeaturesMulti(I_face);
