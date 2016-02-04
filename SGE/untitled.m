

for t = 1067:length(fra_db)
    t
    imgData = fra_db(t);
    segPath = j2m('~/storage/fra_db_mouth_seg_2',imgData);
    load(segPath);
    % correct train, if needed
    if any(~[segs.success])
        'Aha!'
        delete(segPath);
    end
end