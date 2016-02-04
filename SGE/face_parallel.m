function face_parallel(baseDir,d,indRange,outDir)

cd ~/code/mircs;
initpath;
config;
conf.get_full_image = false;

images = {};
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    try
        I = im2uint8(getImage(conf,currentID));
        images{k} = I;
    catch me
        images{k} = imread(fullfile(baseDir,currentID));
    end
end

cd /home/amirro/code/3rdparty/voc-release5
for k = 1:10
    load(sprintf('models/face_big_%d_final.mat',k));
    models(k) = model;
end

startup;


for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,d(indRange(k)).name);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    %     fprintf('checking if results for image %s exist...',filename);
    if (exist(resFileName,'file'))
        fprintf('results exist. skipping\n');
        continue;
    else
        fprintf('calculating...');
    end
    
    im = images{k};
    im = imresize(im,2,'bilinear');
    res = struct('ds',{});
    for iModel = 1:length(models)
        
        dss = {};
        for iRot = -60:10:60
            curDS = detectRotated(im,models(iModel),-1,iRot);
            if (~isempty(curDS))
                curDS = [curDS,repmat(iRot,size(curDS,1),1)];
                dss{end+1} = curDS;
            end
        end
        ds = cat(1,dss{:});
        res(iModel).ds = ds;
    end
    save(resFileName,'res');
end

% fprintf('\n\n\nFINISHED\n\n\n!\n');

% % % function ds = detectRotated(im,model,thresh,theta)
% % % ds = [];
% % % im_orig = im;
% % % sz = size(im);
% % % sz = sz(1:2);
% % % r = ceil(sum(sz.^2)^.5); % side of new image
% % % padSize = ceil([max(r-size(im,1),0),max(r-size(im,2),0)]/2);
% % % % theta = 20
% % % im = padarray(im_orig,padSize,0,'both');
% % % im = imrotate(im,theta,'bilinear','crop');
% % % [ds, bs] = imgdetect(im, model,-1.5);
% % % top = nms(ds, 0.5);
% % % ds = ds(top,:);
% % % if (isempty(ds))
% % %     return;
% % % end
% % % % if ~isempty(ds)
% % % %     ds(:,1:4) = ds(:,1:4)/2;
% % % % end
% % % % ds_orig = ds;
% % %
% % % % ds = ds_orig;
% % % R = rotationMatrix( theta*pi/180 );
% % %
% % % imCenter = size(im);
% % % imCenter = fliplr(imCenter(1:2)/2);
% % %
% % % bc = (ds(:,1:2)+ds(:,3:4))/2;
% % % dd = bsxfun(@minus,bc,imCenter);
% % % rr = (R*dd')';
% % % rr = bsxfun(@plus,rr,imCenter);
% % % ds(:,1:4) = ds(:,1:4)-[bc bc];
% % % ds(:,1:4) = ds(:,1:4)+[rr rr];
% % % ds(:,1:4) = bsxfun(@minus,ds(:,1:4),padSize([2 1 2 1]));
% % %
% % % % showboxes(im,ds);
% % % % figure,showboxes(im,ds_orig)
% % %
% % % % i1 = imrotate(im,-theta,'bilinear','crop');
% % % % i1 = i1(padSize+1:end-padSize,padSize+1:end-padSize,:);
% % % % figure,showboxes(i1,ds)
% % % % figure,showboxes(im_orig,[])