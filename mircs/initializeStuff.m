if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
    %addpath('/home/amirro/code/3rdparty/edgeBoxes/');
    addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
    % rmpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta12/'));
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta14/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta14/examples');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta14/matlab');
%     vl_compilenn('enableGpu',true);
    vl_setupnn
    ensuredir('~/code/mircs/logs');
    addpath('~/code/3rdparty/MinBoundSuite/');
    %%    
    addpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained');
    install
    %%        
%     fra_db = initialize_fra_db(conf);
    load fra_db_2015_10_08
    
    
 
    
    
    %%
%     roiParams = defaultROIParams();        
%     fra_db = fra_db_d;
%      for t = 1:length(fra_db_d)
%         imgData = fra_db(t);
%         clf;
%         conf.get_full_image = true;
% %         imagesc2(getImage(conf,imgData));
%         [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
% %         plotBoxes(imgData.faceBox);
% %         get_rois_fra
%         imagesc2(I);
%         plotPolygons(imgData.face_landmarks.xy(imgData.face_landmarks.valids',:),'g+');
% %         bbPath = j2m(annoPath,imgData,'.jpg.txt');
% %         bb = [];
% %         obj = bbGt('bbLoad',bbPath);
% %         if (~isempty(obj))
% %             bb = cat(1,obj.bb);
% %             bb = bb(:,1:4);
% %             bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
% %         end
% %         fra_db(t).hands = bb;
% dpc
% 
%     end
   %     
    %bbLabeler({'hand'},myTmpDir,myTmpDir);
    % bbLabeler({'hand'},conf.imgDir,'/home/amirro/data/Stanford40/annotations/hands/');
    % update hands!
    annoPath = '/home/amirro/storage/data/Stanford40/annotations/hands';
    annoPath_orig = '/home/amirro/data/Stanford40/annotations/hands/';
    addpath('/home/amirro/code/3rdparty/gaimc');
    % bbLabeler({'hand'},conf.imgDir,annoPath)
    listOfNeededPaths = {};
    for t = 1:length(fra_db)
        imgData = fra_db(t);
        bbPath = j2m(annoPath,imgData,'.jpg.txt');
        bb = [];
        obj = bbGt('bbLoad',bbPath);
        if (~isempty(obj))
            bb = cat(1,obj.bb);
            bb = bb(:,1:4);
            bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
        end
        fra_db(t).hands = bb;
    end
    
    % r=prepare_dlib_data(conf,fra_db,'dlib_input.txt');
    % (run landmarks... )
    % load the landmark detection
    fra_db = load_dlib_landmarks(conf,fra_db,'dlib_input.txt','~/dlib_output.txt',false);            
    % fra_db = load_dlib_landmarks(conf,fra_db,'dlib_input.txt','~/dlib_output_profile.txt',true);        
    %for t = [680   690   700   710   720   730   740   750]    %     
     
    %%
        
    %%
    params = defaultPipelineParams();
    %% TRAINING
    % 1. Define a graph structure
    nodes = struct('name',{},'type',{},'spec',{},'params',{},'bbox',{},'poly',{},'valid',{});
    edges = struct('v',{},'connection',{});
    isTrain = [fra_db.isTrain];
    posClass = 4; % brushing teeth
    isClass = [fra_db.classID] == posClass;
    isValid = true(size(fra_db));%[fra_db.isValid];
    % findImageIndex(fra_db,'brushing_teeth_064.jpg')
    train_pos = isClass & isTrain & isValid;
    train_neg = ~isClass & isTrain & isValid;
    f_train_pos = find(train_pos);
    f_train_neg = find(train_neg);
    test_pos = isClass & ~isTrain & isValid;
    test_neg = ~isClass & ~isTrain & isValid;
    f_test_pos = find(test_pos);
    f_test_neg = find(test_neg);
    f_train = find(isTrain & isValid);
    f_test = find(~isTrain & isValid);
    % check the coverage of the ground-truth regions using edgeboxes.
    % clf;
    % 1. define graph-structure
    % 2. define a way to transform graph to image
    % 3. extract features from graph
    % 4. optimize cost function over graph.
    % Define a graph, whos nodes are:
    % mouth, object, hand
    % interaction between nodes is important
    % the state of a node may by not only location, but also orientation
    nodes(1).name = 'mouth';
    nodes(1).type = 'region';
    nodes(1).spec.size = .25;
    nodes(2).name = 'obj';
    nodes(2).type = 'poly';
    nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
    nodes(3).name = 'hand';
    nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
    nodes(3).type = 'region';
    %state of a node may by not only location, but also orientation
    nodes(1).name = 'mouth';
    nodes(1).type = 'region';
    nodes(1).spec.size = .25;
    nodes(2).name = 'obj';
    nodes(2).type = 'poly';
    nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
    nodes(3).name = 'hand';
    nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
    nodes(3).type = 'region';
    
    edges(1).v = [1 2];
    edges(1).connection = 'anchor';
    edges(2).v = [2 3];
    edges(2).connection = 'anchor';
    for iNode = 1:length(nodes)
        nodes(iNode).valid = true;
    end
    % sample good configurations, configurations are assignments to the graph,
    % and each configuration is defined a set of polygons defining image
    % regionsate of a node may by not only location, but also orientation
    nodes(1).name = 'mouth';
    nodes(1).type = 'region';
    nodes(1).spec.size = .25;
    nodes(2).name = 'obj';
    nodes(2).type = 'poly';
    nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
    nodes(3).name = 'hand';
    nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
    nodes(3).type = 'region';
    
    %
    % let's do this in several stages, with differing levels of complexity.
    % The regions can be: bounding boxes or oriented polygons
    % The geometric relation between the nodes can be constrained or
    % non-constrained
    % The node samples can be either any region or regions derived from some
    % proposal method (e.g) edge boxes
    configurations = {};
    params.gt_mode = 'box_from_poly';
    params.rotate_windows = true;
    params.cand_mode = 'polygons';
    params.cand_mode = 'boxes';
    params.cand_mode = 'segments';
    params.feature_extraction_mode = 'bbox';
    params.holistic_features = false; %TODO - remeber to check this as a baseline...
    params.interaction_features = 1;
    regionSampler = RegionSampler();
    regionSampler.clearRoi();
    nSamples = 20;
    theta_start = 0;
    theta_end = theta_start+360;
    b = (theta_end-theta_start)/nSamples;
    theta_end = theta_end-b;
    thetas = theta_start:b:theta_end;%0:10:350;
    lengths = .5;
    widths = .5;
    params.sampling.thetas = theta_start:b:theta_end;%0:10:350;
    params.sampling.lengths = lengths;
    params.sampling.widths = widths;
    params.sampling.nBoxThetas = 8;
    %params.sampling.boxSize = [.5 .7 1];
    params.sampling.boxSize = [.5 .7 1];
    params.sampling.maxThetaDiff = 50;
    %[.3 .5 .7];
    
    
    
    %%
    featureExtractor = DeepFeatureExtractor(conf,true);
    initialized = true;
end
