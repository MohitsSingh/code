imgDir = '/home/amirro/storage/datasets/image_net/images_unpacked/';
% d = dir(fullfile(imgDir,catDir,'*.JPEG'));
prms.cell_size = cell_size;
prms.features = features;
prms.detection = detection;
threshold = 0;
prms.threshold = threshold;
% 1. load the records for each class.
% 2. detect on them the desired class.
% 3. return the bounding boxes of the highest scoring detections.
% 4. train again.
if (0)
    load all_all_rects;
    all_all_rects = {};
    newRecs = {};
    for k = 1:length(classifiers)
        k
        curClassifier = classifiers(k);
        curClass = classifiers(k).class;
        curRecs = getRecsForClass(recs,curClass);
        %         all_rects = detectOnImageSet(curClassifier,imgDir,curRecs,prms);
        all_rects = all_all_rects{k};
        all_all_rects{k} = all_rects;
        newRecs{k} = select_good_detections(all_rects,curRecs,curClass,200,imgDir);
    end
    save all_all_rects all_all_rects
    recs = col(cat(2,newRecs{:}));
end
extra.suffix = '_refined';
negativeDir = '/net/mraid11/export/data/amirro/applauding';
dd = dir(fullfile(negativeDir,'*.jpg'));
negatives= {};
for k = 1:length(dd), negatives{k} = fullfile(negativeDir,dd(k).name); end
extra.negativeImages = negatives;
% override the VOCopts to match the imagenet data.
run_circulant_custom
wnids = {synsets.wnid};
% match each classifier with a name
for k = 1:length(classifiers)
    f = find(cellfun(@any,strfind(wnids,classifiers(k).class)));
    classifiers(k).name = synsets(f).name;
end
