ethzDir = '/home/amirro/ETHZShapeClasses-V1.2/';
recDir = 'obj_records';
object_types = {'Bottles','Mugs'};

addpath('/home/amirro/data/VOCdevkit/VOCcode/');

nRec = 0;

resizeFactor = .5;
for k = 1:length(object_types)
    % get files for this object
    obj_images = dir(fullfile(ethzDir,object_types{k},'*.jpg'));
    
    % get for each file the ground truth bounding box,
    % create a pascal record for each image.
    
    for iImage = 1:length(obj_images)
        iImage
        imgname_orig = fullfile(ethzDir,object_types{k},obj_images(iImage).name);
        if (~isempty(strfind(imgname_orig,'_small.jpg')))
            continue;
        end
        I = imread(imgname_orig);
        imgname = strrep(imgname_orig,'.jpg','_small.jpg');
        I = imresize(I,resizeFactor);
        imwrite(I,imgname);
        imgsize = size(imread(imgname));
        database = 'ethz';
        rec = PASemptyrecord;
        nRec = nRec + 1;
        % read bounding boxes. this is provided as xmin ymin xmax ymax
        m = dlmread(strrep(imgname_orig,'.jpg',['_' lower(object_types{k}) '.groundtruth']));
        objects = [];
        for iRect = 1:size(m,1)
            obj = PASemptyobject;
            obj.bbox = round(resizeFactor*m(iRect,:));
            obj.label = object_types{k};
            obj.class = object_types{k};
            objects = [objects;obj];             %#ok<AGROW>
        end
        
        %         rec.imgname = imgname;
        %         rec.imgsize=  imgsize;
        %         rec.database = database;
        %         rec.objects = objects;
        
        save(fullfile(recDir,sprintf('%03.0f.mat',nRec)),...
            'imgname','imgsize','database','objects');
    end
end