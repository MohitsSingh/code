function imdb = getIMDB(imdb_type)
% image database; for each image there may be multiple objects
% for which the action can be identified.
imdbRoot = 'data/imdb';
ensuredir(imdbRoot);
switch imdb_type
    case 'stanford_40'
        imdbPath = fullfile(imdbRoot,'stanford_40_imdb.mat');
        if (exist(imdbPath,'file'))
            load(imdbPath);
        else
            %             baseDir = '/home/amirro/storage/data/Stanford40/';
            %             annoDir = fullfile(baseDir,'MatlabAnnotations');
            %             imageSplitDir = fullfile(baseDir,'ImageSplit');
            %             imgDir = fullfile(baseDir,'JPEGImages');
            %             all_ids = [train_ids;test_ids];
            %             all_labels = [train_labels;test_labels];
            %             imdb = struct('isTrain','image_id',{},'image_path',{},'objects',{});
            %             for t = 1:length(train_ids)
            %                 ischar_ = 1;
            %                 [~,fname,~] = fileparts(I);
            %                 xml_path = fullfile(conf.xmlDir,[fname '.xml']);
            %                 a = loadXML(xml_path);
            %                 xmin = str2num(a.annotation.object.bndbox.xmin);
            %                 xmax = str2num(a.annotation.object.bndbox.xmax);
            %                 ymin = str2num(a.annotation.object.bndbox.ymin);
            %                 ymax = str2num(a.annotation.object.bndbox.ymax);
            %                 %         end
            %             end
            %             save(imdbPath,'imdb');
        end
    case 'PASCAL2012'
        imdbPath = fullfile(imdbRoot,'pascal2012_imdb.mat');
        if (exist(imdbPath,'file'))
            load(imdbPath);
        else
            imdb = get_action_data();
            for u = 1:length(imdb)
                imdb(u).idx = u;
            end
            save(imdbPath,'imdb');
        end
end
end