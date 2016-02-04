function [qq,q,aps] = applyToSet(conf,clusters,imgSet,labels,suffix,varargin)
%APPLYTOSET
%   applyToSet(conf,clusters,imgSet,labels,suffix,varargin)

ip = inputParser;
ip.addParamValue('toSave',true,@islogical);
ip.addParamValue('nDetsPerCluster',10,@isnumeric);
ip.addParamValue('add_border',false,@islogical);
ip.addParamValue('useLocation',0);
ip.addParamValue('perImageMasks',0);
ip.addParamValue('disp_model',false,@islogical);
ip.addParamValue('override',false,@islogical);
ip.addParamValue('dets',[]);
ip.addParamValue('visualizeClusters',true,@islogical);
ip.addParamValue('uniqueImages',true,@islogical);
ip.addParamValue('rotations',0);
ip.addParamValue('sals',{},@iscell);
ip.parse(varargin{:});

if (isempty(labels))
    labels = col(true(size(imgSet)));
end

if (~isempty(ip.Results.dets))
    q =ip.Results.dets;
else
    conf.detection.params.detect_keep_threshold = -1000;
    conf.detection.params.detect_max_windows_per_exemplar = 10;
    
    %     tic
    %     [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,imgSet{1},1 );
    %     dd = toc
    
    q = getDetections(conf,imgSet,clusters,[],suffix,ip.Results.toSave,...
        ip.Results.override,ip.Results.rotations);
    %     dd = toc
end

qq = getTopDetections(conf,q,clusters,'uniqueImages',ip.Results.uniqueImages,...
    'useLocation',ip.Results.useLocation,...
    'nDets',inf,'sals',ip.Results.sals,'perImageMasks',ip.Results.perImageMasks);
[prec,rec,aps,T,M] = calc_aps(qq,labels);
% disp(aps)
if (ip.Results.visualizeClusters)
    [A,AA] = visualizeClusters(conf,imgSet,qq,'add_border',...
        ip.Results.add_border,'nDetsPerCluster',...
        ip.Results.nDetsPerCluster,'gt_labels',labels,...
        'disp_model',ip.Results.disp_model,'height',64);
    imwrite([clusters2Images(A)],[suffix '.jpg']);
    %     imshow(multiImage(AA,true,false));
    %clf; imagesc([clusters2Images(A)]);axis image;
end
end