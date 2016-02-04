if (1)
    
    inputDir = '/home/amirro/data/Stanford40/JPEGImages';
    ext = '.jpg';
    
    actionsFileName = '/home/amirro/data/Stanford40/ImageSplits/actions.txt';
    [A,ii] = textread(actionsFileName,'%s %s');
    
    f = fopen(actionsFileName);
    A = A(2:end);
    
    for k =[2 3 9 24 32]
        k
        currentTheme = A{k};
        currentTheme
        hands_locs_dir = fullfile(inputDir,[currentTheme '_' opts.hands_locs_suff]);
        prefix = currentTheme;
        if (~exist(hands_locs_dir,'dir'))
            mkdir(hands_locs_dir);
        end
        simple_labeler(inputDir,prefix,ext,hands_locs_dir);
    end
    %%
    
    for k = 9:length(A)
        k
        currentTheme = A{k};
        hands_imgs_dir = fullfile(inputDir,[currentTheme '_' opts.hands_images_suff]);
        hands_locs_dir = fullfile(inputDir,[currentTheme '_' opts.hands_locs_suff]);
        prefix = currentTheme;
        if (~exist(hands_imgs_dir,'dir'))
            mkdir(hands_imgs_dir);
        end
        cutter(inputDir,prefix,ext,hands_locs_dir,hands_imgs_dir);
    end
end

if (0)
    % correct the pascal records according to the sorted hand images
    prefix = 'drinking';
    rec_dir = fullfile(inputDir,'drinking_hands_locs');
    sortedDirBase = '/home/amirro/data/Stanford40/sorted';
    sortedDirs = dir(sortedDirBase);
    rec_files = dir(fullfile(rec_dir,'*rec.mat'));
    for iRec = 1:length(rec_files)
        iRec
        rec = load(fullfile(rec_dir,rec_files(iRec).name));
        for iObj = 1:length(rec.objects)
            curObj = rec.objects(iObj);
            if (any(strfind(curObj.class,'?')))
                continue;
            end
            for iDir = 1:length(sortedDirs)
                if (length(sortedDirs(iDir).name) < 3)
                    continue;
                end
                                
                d = dir(fullfile(sortedDirBase,sortedDirs(iDir).name,...
                    sprintf('%s_%05.0f.jpg',prefix,curObj.label)));
                if (~isempty(d))
                    curObj.class = [curObj.class '_' sortedDirs(iDir).name];
                    rec.objects(iObj) = curObj;
                    break;
                end
            end
        end
        imgname = rec.imgname; %#ok<*NASGU>
        imgsize = rec.imgsize;
        objects = rec.objects;
        database = rec.database;
        save(fullfile(rec_dir,rec_files(iRec).name),'imgname','imgsize','database','objects');
    end
    
    
    
end

% split to 1 image per object using bounding boxes
    data_dir = 'drinking_dir';
    if (~exist(data_dir,'dir'))
        mkdir(data_dir);
    end
count_  = 0;
c = 0;
for iRec = 1:length(rec_files)
    iRec
    rec = load(fullfile(rec_dir,rec_files(iRec).name));
    im = imread(rec.imgname);
    database = rec.database;
    for iObj = 1:length(rec.objects)
        curObj = rec.objects(iObj);
        bbox = curObj.bbox;
        if (isempty(bbox))
            continue;
        end
        count_ = count_ + 1;
        ddd = 80;
        xmin = max(1,bbox(1)-ddd);
        ymin = max(1,bbox(2)-ddd);
        xmax = min(size(im,2),bbox(3)+ddd);
        ymax = min(size(im,1),bbox(4)+ddd);
        %cur_im = im(bbox(2):bbox(4),bbox(1):bbox(3),:);
        cur_im = im(ymin:ymax,xmin:xmax,:);
        cur_im_resized = cur_im;
        %cur_im_resized = standardizeImage(cur_im,64);
        
        size_ratio = size(cur_im_resized,1)/size(cur_im,1);
        cur_bbox = [bbox(1)-xmin,bbox(2)-ymin,bbox(3)-xmin,bbox(4)-ymin];
        cur_bbox = round(cur_bbox*size_ratio);
        cur_im = cur_im_resized;
        %cur_bbox = [1 1 size(cur_im,2) size(cur_im,1)];
        obj = PASemptyobject;
        obj.bbox = cur_bbox;
        obj.class = curObj.class;
%         if (~strcmp(curObj.class,'drinking_bottle'))
%             continue
%             c = c+1;
%         end
        obj.view = curObj.view;
        obj.label = count_;
        
        recFile = fullfile(data_dir,sprintf('%04.0f.mat',count_));
        imgFile = fullfile(data_dir,sprintf('%04.0f.jpg',count_));
        
        imgname = fullfile(pwd,imgFile);
        imgsize = size(cur_im);
        objects = obj;
        save(recFile,'imgname','imgsize','database','objects');
        imwrite(cur_im,imgFile);
    end
end
%
% cc= 0;
% for kk = 1:256
%     recFile = fullfile(data_dir,sprintf('%04.0f.mat',kk));
%     rec = load(recFile);
%     if (strcmp(rec.objects(1).class,'drinking_bottle'))
%         cc = cc+1
%     end
% end
%

% TODO: now create a new directory, called "test",
% and write out cropped images , containing only the hands,with corresponding records,
% so we can test the felzenszwalb as a classifier , not as a detector.



% prefix = 'texting_message';
% outDir = 'D:\Stanford40\texting_message_locs';
% if (~exist(outDir))
%     mkdir(outDir);
% end6
% simple_labeler(inputDir,prefix,ext,outDir);


% now choose the subset of hands holding things....
% queryString = {'grasping','empty'};
% asker(inputDir,prefix,ext,outDir,queryString);
%