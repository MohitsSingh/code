
%%
% conf.VOCopts = VOCopts;
conf.pasMode = 'none';

%% some constants
conf.consts.MODEL = 6;
conf.consts.SCORE = 12;
conf.consts.SCALE = 8;
conf.consts.FLIP = 7;
conf.demo_mode = 0;

conf.classes = A;

% create a class enum....
conf.class_enum = [];
for k = 1:length(conf.classes)
    eval(sprintf('conf.class_enum.%s=%d;',upper(conf.classes{k}),k));
end

conf.class_subset = conf.class_enum.DRINKING;

%%
conf.DPM_path = '/home/amirro/code/3rdparty/voc-release4.01/';

%% some directories
conf.cachedir = '~/storage/data/cache';
conf.demodir = '~/storage/data/demo';
conf.gpbDir = '~/storage/gpb_s40';
conf.gpbDir_face = '~/storage/gpb_s40_face';
conf.lineSegDir = '~/storage/lineseg_s40';
conf.bowDir = '~/storage/bow_s40';
conf.handsDir = '~/storage/hands_s40';
conf.classificationDir = '~/storage/res_s40';
conf.dpmDir= '~/storage/dpm_s40';
conf.shapeFeatsPath = '~/storage/shape_s40';
conf.annotationPath = '/home/amirro/storage/data/Stanford40/annotations/JPEGImages';
% conf.occludersDir = '~/storage/occluders_s40';
conf.occludersDir = '/home/amirro/storage/occluders_s40_new4';
conf.elsdDir_gpb = '~/storage/s40_elsd_gpb';
conf.elsdDir = '~/storage/s40_elsd_output/';
conf.upperBodyDir = '~/storage/upper_bodies_s40';
conf.segDataDir = '/home/amirro/storage/s40_seg_data';
conf.landmarks_piotrDir = '~/storage/s40_keypoints_piotr';
conf.landmarks_myGTDir = '~/storage/all_kp_preds_new';
conf.landmarks_myDir = '~/storage/s40_my_facial_landmarks';
conf.face_seg_dir = '~/storage/s40_fra_face_seg';
conf.saliencyDir = '~/storage/s40_sal_fine';
conf.action_pred_dir = '~/storage/s40_action_pred';

ensuredir(conf.cachedir);
ensuredir(conf.demodir);
%% clustering parameters
conf.clustering = struct;
conf.clustering.windows_per_image = 150;
conf.clustering.sample_method = 'quant_error';

% sample the data at a large ratio (for debugging);
conf.clustering.set_sampling = 1; % was 3
% minimal HOG gradient energy per window, used to remove patches
% with no informative features (although this is questionable,
% as sky can be informative to some classes).
% this refers actually to mean per-bin energy,
% so we won't have to set a different threshold for each window size.
conf.clustering.min_hog_energy = .2;

conf.clustering.split_discovery = true; % split training set for train/validation?

conf.clustering.cluster_ratio = 4; % ratio of clusters to number of data points
conf.clustering.num_iter = 5;
% number of detections to retain after each round of svm detections.
conf.clustering.top_k = 5;
% minimal number of elements to retain for a cluster before throwing
% it away
conf.clustering.min_cluster_size = 3;

conf.debug.cluster_choice = [];
conf.debug.override = true;

conf.clustering.num_hard_mining_iters = 5;

conf.clustering.max_sample_ovp = .5; %TODO - this will be fixed
% when clustering, since no cluster will be allowed to contain two patches
% from the same image.
% maximal image size, images above this size will be resized.
conf.max_image_size =inf;
conf.get_full_image = true;

%% detector parameters
% conf.detection.params = esvm_get_default_params;
% conf.detection.params.max_models_before_block_method = 10;
% conf.detection.params.init_params.sbin = 8;
% conf.detection.params.detect_levels_per_octave = 8; % was 4
% conf.detection.params.detect_min_scale = .5; % this will result in about 7 scales. (TODO - change
% % this back to .2, or it's not really 7 scales...);
% conf.detection.params.detect_save_features = false;
% conf.detection.params.detect_add_flip = 1; % TODO - this was added for debugging,
% % but make sure that boxes2features works also when it's turned on.
% conf.detection.params.detect_pyramid_padding = 0; % don't use any padding for now...
% % should other cluster's samples serve as negatives for the first phase
% % of training w.r.t to the trained cluster?
% conf.clustering.split_clusters = false;
% 
% conf.detection.params.init_params.hg_size = [8 8];
% conf.detection.params.init_params.MAXDIM = 8;


%conf.suffix = '_debug';

%% secondary clustering parameters
conf.clustering.secondary.img_size = 128;
conf.clustering.secondary.inflate_factor = 1.5;
conf.clustering.secondary.sample_size = 100;

conf.level = 1;

%% bag-of-words learning params.
conf.bow.maxNegatives_debug = 100;
% conf.bow.tiling = [1 2];
conf.bow.tiling = [1 2];
conf.bow.imagesPerRound_debug = 3;

conf.bow.maxNegatives = 2000;
conf.bow.imagesPerRound = 5;

%% parallel (SGE)
conf.parallel = false;

%% HOG feature configuration
conf.features = struct;
conf.features.winsize = [8 8]; % in bins, e.g, 64x64 pixels (really?

conf.features.vlfeat.cellsize = 8;
% conf.detection.params.init_params.features = @piotr_features;
% conf.features.fun = conf.detection.params.init_params.features;

% occlusion options
conf.occlusion.whatFace = 'landmark';
% conf.occlusion.whatFace = 'seg';

%% straw related configuration and more
conf.straw.extent = 1;
conf.straw.dim = 100;
n = 1;
conf.straw.primitive_opts(n).method = 'canny';
conf.straw.primitive_opts(n).filter = []; n=n+1;
conf.straw.primitive_opts(n).method = 'canny';
conf.straw.primitive_opts(n).filter = fspecial('gauss',7,3); n=n+1;
conf.straw.primitive_opts(n).method = 'canny';
conf.straw.primitive_opts(n).filter = fspecial('gauss',9,5); n=n+1;
conf.straw.primitive_opts(n).method = 'gpb';
conf.straw.primitive_opts(n).filter = []; n=n+1;

% data related to piotr's coordinates
conf.piotr_coords.eye_left = [1 3 5 6 9 11 13 14 17];
conf.piotr_coords.eye_right = [2 4 7 8 10 12 15 16 18];
conf.piotr_coords.nose = 19:22;
conf.piotr_coords.mouth = 23:28;
conf.piotr_coords.chin = 29;
conf.piotr_coords.mouth_left_corner = 23;
conf.piotr_coords.mouth_right_corner = 24;
conf.piotr_coords.mouth_top_inner = 26;
conf.piotr_coords.mouth_bottom_inner = 27;
