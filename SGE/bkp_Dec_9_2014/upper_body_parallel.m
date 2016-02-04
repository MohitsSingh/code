function upper_body_parallel(baseDir,d,indRange,outDir)

dpmPath = '~/code/3rdparty/voc-release5/';
addpath(genpath('~/code/utils'));
addpath(genpath(dpmPath));
startup;
load /home/amirro/code/3rdparty/disc_subcat/ubDetModel.mat

%
for k = 1:length(indRange)
    imgPath = d(indRange(k)).name;
    
    if (strcmp(ext,'.txt')), continue, end;
    loadOrCalc([],@rcpr_image,imread(imgPath),j2m(outDir,imgPath));
end

    function res = rcpr_image(conf,I,varargin)
        %     function res = rcpr_image(conf,I,regModel,bboxesTr,regPrm,prunePrm)
        regModel = varargin{1}{1};
        bboxesTr = varargin{1}{2};
        regPrm = varargin{1}{3};
        prunePrm = varargin{1}{4};
        [res] = detect_on_set({I},regModel,bboxesTr,regPrm,prunePrm);
        res = res{1};
    end
end
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,currentID);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    
    I = imread(imagePath);
    res = imgdetect(I,model,-1.1);
    save(resFileName,'res');
    
    fprintf('done with image %s!\n',filename);
end
fprintf('\n\n\finished all images in current batch\n\n\n!\n');
end

