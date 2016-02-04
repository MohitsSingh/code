
VOCinit;

[imgids,objids]=textread(sprintf(VOCopts.layout.imgsetpath,VOCopts.trainset),'%s %d');
n = 0;
close all
tic

d = '/home/amirro/code/handson/hands';
mkdir(d);
for i=1:length(imgids)
    % display progress
    if toc>1
        fprintf('train: %d/%d\n',i,length(imgids));
        drawnow;
        tic;
    end
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,imgids{i}));
    
    % extract object
    
    %     objects(n)=rec.objects(objids(i));
    
    I = imread(sprintf(VOCopts.imgpath,imgids{i}));
    
    for k = 1:length(rec.objects)
        obj = rec.objects(k);
        p = obj.part;
        for kk = 1:length(p)
            prt = p(kk);
            if (strcmp(prt.class,'hand') == 1)
                n=n+1
                bbox = prt.bbox;
                bbox = inflatebbox(bbox,1);
                bbox(3:4) = bbox(3:4)-bbox(1:2);
                I2 = imcrop(I, bbox);
                I2 = imresize(I2,[128 NaN]);
                imwrite(I2,fullfile(d,[num2str(n,5) '.jpg']));
%                 imshow(I2);
%                 pause(.1);
            end
        end
    end
    
    %     % move bounding box to origin
    %     xmin=objects(n).bbox(1);
    %     ymin=objects(n).bbox(2);
    %     objects(n).bbox=objects(n).bbox-[xmin ymin xmin ymin];
    %     for j=1:numel(objects(n).part)
    %         objects(n).part(j).bbox=objects(n).part(j).bbox-[xmin ymin xmin ymin];
    %     end
end