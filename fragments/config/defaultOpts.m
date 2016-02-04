% initpath;
VOCinit; % PASCAL toolkit
vl_setup; % vl_fleat

% some configuration parameters...
globalOpts.dataDir = 'data';
globalOpts.training_set = 'train';
globalOpts.testing_set = 'val';
globalOpts.presmooth = false;

% Use bounding boxes from segmentation (0) or resize them? (this
% is the sqrt of the bbox area, so 128 is actualy square of area 128*128)
globalOpts.det_rescale = 0;

globalOpts.minBboxArea = 25; % minimal accepted candidate bounding box

globalOpts.scale_choice = [];
globalOpts.useGistFeatures = 0;

globalOpts.aib_cut = 256;

globalOpts.hkmfun = @hkm; % don't use anything (as opposed to homogeneous kernel maps)

% Which classes to train on? Use only for debugging, don't change this
globalOpts.class_subset = [1:20];

globalOpts.trainRange = [0 1]; %obsolete
globalOpts.testRange = [0 1]; %obsolete

globalOpts.numTrain = 50; % number of train & test bboxes used for each category.
% numTrain positives and numTrain negatives are sampled from the data.
globalOpts.maxTrain = 300; % maximal number of samples (postive / negative)

globalOpts.numTest = 50;  % obsolete
globalOpts.descfun = @phowDesc; % handle to descriptor extraction function
% globalOpts.phowOpts = {'Step', 1, 'Sizes', [4],'Fast',1,'FloatDescriptors',1};
globalOpts.phowOpts = {'Step', 1, 'Sizes', [2 4 6 8 10],'Fast',1,'FloatDescriptors',1};
globalOpts.phowOpts_sample = {'Step', 20, 'Sizes', 4,'Fast',1,'FloatDescriptors',1};

% globalOpts.descfun = @dense_sift;

globalOpts.int_area_thresh = .5; % use for sampling positives using segmentation

globalOpts.use_overlapping_negatives = true;

globalOpts.partial_det = false;

% rather than ground truth bounding boxes
globalOpts.numWords = 4096;

globalOpts.keepDescFiles = false; % delete descriptor files after used.
% you should probably leave this false, since it will fill your storage
% very quickly

globalOpts.learn.useGTBoxes = 1;
globalOpts.debug = 0; % set to true for visualization, etc.

% parameters of spatial pyramid
globalOpts.numSpatialX = [2 4];
globalOpts.numSpatialY = [2 4];

globalOpts.VOCopts = VOCopts;
