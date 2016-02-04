function detect_kas_exec_interface(d, image_type, vak)

% executable interface for kas detection software
%
% pre-processes all images in directory d,
% computing edgemaps, edgel-chains, contour segment network, and kas
%
% If vak not given -> stop at building csn
%

% process arguments
if nargin < 3
  vak = -1;
end

if nargin < 2
  disp('Usage:');
  disp([mfilename ' directory image_type [k]']);
  newline;
  disp('  directory:  where the images to be processed are');
  disp('  image_type: image file extension, e.g. jpg');
  disp('  k:          degree of kAS features to detect.');
  return;
end

% attempt to add path to unitex*.mat needed by Berkeley edge detector
bup = getenv('BERKELEY_UNITEX_PATH');
if isempty(bup)
  display('WARNING: BERKELEY_UNITEX_PATH environment variable not set.');
  display('Berkeley edge detector might not work properly.');
  display('Please set BERKELEY_UNITEX_PATH to where your unitex*.mat files are.');
else
  addpath(bup);
end

% matlab compiled executables get strings passed from the shell
if ischar(vak)
  vak = str2num(vak);
end

% edge detection parameters
preprocess_params.edge_detector = 'berkeley';
preprocess_params.hysteresis = [0 0.1];

% main processing
PreprocessImagesDir(d, image_type, preprocess_params, vak);
