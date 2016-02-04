function kasPrimitives = getKASPrimitives(conf,imageID)
    fName = fullfile(conf.kasPrimitivePath,strrep(imageID,'.jpg','.mat'));
    if (exist(fName,'file'))
        load(fName);
    end
    error(['primitives for file ' imageID ' don''t exist']);
end