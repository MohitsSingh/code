function PreprocessImagesDir(d, image_type, params, vak)

% pre-processes all images in directory d,
% thereby preparing all data necessary at recognition time.
% Saves .mat file with processed images.
%
% If vak given -> compute also kAS, and save them both as .pas, .pas_ls fields of image.mat,
% and as image.segments, image.<vak>AS ascii files. If image.mat already exists,
% its .pas and .pas_ls fields are overwritten.
%

flipping = false;                     % horiz flip edgemaps ?

if nargin < 4
  vak = -1;
end

global MY_TOOLS;                      % to invoke VxL's Canny edge detector

% save current working directory
org_dir = pwd;

% get list of images
cd(d);
images = dir(['*.' image_type]);
disp('Images found: ');
dir(['*.' image_type]);               % must display this way, because variable is a struct
newline;


% preprocess images
for i = images'
  disp(i.name);
  disp('--------------------');

  % shortcuts
  name = i.name;
  i_base = name(1:(length(name)-length(image_type)-1));
  edges_fname = [i_base '_edges.tif'];
  chains_fname = [i_base '.chains'];
  mat_fname = [i_base '.mat'];
  segments_fname = [i_base '.segments'];
  kas_fname = [i_base '.' num2str(vak) 'AS'];
  guard_fname = [i_base '.guard'];

  % check for parallel processing
  if exist([pwd '/' guard_fname],'file')           % WARNING: without prepending pwd, exist searches the entire matlab path !
    disp('Guard file found -> another process is treating this image -> skip to the next one');
    continue;
  else
    % write guard to support parallalel processing
    trash=0; save(guard_fname, 'trash');
  end

  % detect edgels
  model_img = false;
  if not(exist([pwd '/' edges_fname], 'file'))
    img = imread(i.name);
    if is_binary(img)
      % for uniformity, want to pass all images through same pipeline (plus, binary images don't get self-crossings ...)
      model_img = true;
      disp('Binary image -> consider as a sketch (without self-crossing).');
      disp('Extracting edgelchains...');
      clear t; t.ecs = DetectEdgelChains(img);
    else % not a binary image
      %
      % detect edgels
      if strcmp(params.edge_detector,'berkeley')
        %
        % compute Berkeley edges
        I = imread(i.name);
        if size(I,3) == 1
          disp('Greyscale image: using standard ''brightness-gradient + texture-gradient'' Berkeley edge detector with default parameters');
          [pb theta] = pbBGTG(double(I)/255);
        else
          disp('Color image: Using standard ''color-gradient + texture-gradient'' Berkeley edge detector with default parameters');
          [pb theta] = pbCGTG(double(I)/255);   % 'theta' = orientation map, used only by Chamfer matcher
        end
        pb = hysteresis(pb, params.hysteresis(1), params.hysteresis(2));
        imwrite(pb, edges_fname, 'tiff');
        %
        % compute theta maps (for oriented Chamfer Matching)
        %theta(pb==0) = 0;
        %theta_fname = [i_base '.orient'];  % save .mat as .orient not to confuse later processing steps (who assume that only img.mat is saved as .mat)
        %save(theta_fname,'theta');
        %continue;  % just want to compute the theta maps !
      else
        % compute Canny edges
        disp('Using VxL greylevel Canny edge detector with default parameters');
        cmd = [ MY_TOOLS '/find_edgel_chains.sh ' MY_TOOLS ' "' i.name '" "' chains_fname '"' ];
        disp('Invoking command:'); disp(cmd); system(cmd);
      end % which edge detector ?
    end % binary image ?
  else
    disp([edges_fname ' found -> not computing edges']);
  end
  
  % flipping
  if flipping
    disp(['flipping edgemap ' edges_fname]);
    trash = imread(edges_fname);
    trash = trash(:,end:-1:1);
    imwrite(trash, edges_fname, 'tiff');
  end
  % end flipping

  % chain edgels
  if not(model_img)
  if not(exist([pwd '/' chains_fname], 'file'))
    if not(exist([pwd '/' mat_fname], 'file'))
      % chain edgels, so that they are ready for building the CSN below
      disp('Chaining edgels');
      clear t; t.ecs = ChainEdgels(imread(edges_fname));  % matlab version: easy to port, faster, and even better results !
    end
  else
    disp([chains_fname ' found -> loading edgelchains ' chains_fname]);
    clear t; t.ecs = LoadEdgelChains(chains_fname);
  end
  end

  % Compute CSN related data
  save_matfile = false;
  if not(exist([pwd '/' mat_fname], 'file')) % || true % || true -> do it anyway ;)
    %t = load(mat_fname); t = t.obj;                  % just when forcing to continue
    % by now t exists, with only one subfield .ecs
    t.name = i_base;
    t.ifname = i.name;
    t = PreprocessImage(t, false, model_img);
    save_matfile = true;
    SaveSegments(t.mainlines, t.strengths, segments_fname);
  else % .mat found
    if vak <= 0  % nothing else to do
      disp([mat_fname ' found -> not fitting contour segments, nor computing contour segment network']);
    else % will need to compute kAS
      disp(['Loading Contour Segment Network data ' mat_fname]);
      t = load(mat_fname); t = t.obj;
    end
  end

  % Compute kAS data
  if vak > 0
    t = AddkASData(t, false, true, vak);
    SavekAS(t.pas, t.pas_ls, t.pas_strengths, kas_fname);  % save in ASCII form
    save_matfile = true;
  end

  % save output CSN and kAS
  if save_matfile
    % Save output
    disp(['Saving ' mat_fname]);
    obj = t; clear t;
    if version('-release') > 13  % version 7.x or higher -> forcing to save in V6 maximises compatibility
      save(mat_fname,'obj','-V6'); 
    else
      save(mat_fname,'obj');     % version 6.x or lower
    end
  end

  % delete guard file
  delete(guard_fname);

  newline;
end

% restore current working directory
cd(org_dir);
