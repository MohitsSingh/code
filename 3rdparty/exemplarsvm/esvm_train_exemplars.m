function [newmodels,new_models_name] = ...
    esvm_train_exemplars(models, train_set, params)
% Train models with hard negatives mined from train_set
% [models]: a cell array of initialized exemplar models
% [train_set]: a virtual set of images to mine from
% [params]: localization and training parameters

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

if length(models) == 0
  newmodels = models;
  new_models_name = '';
  return;
end

if length(params.dataset_params.localdir)==0
  CACHE_FILE = 0;
else
  CACHE_FILE = 1;
end

models_name = models{1}.models_name;
new_models_name = [models_name params.training_function()];

cache_dir =  ...
    sprintf('%s/models/',params.dataset_params.localdir);

cache_file = ...
    sprintf('%s/%s.mat',cache_dir,new_models_name);

cache_file_stripped = ...
    sprintf('%s/%s-stripped.mat',cache_dir,new_models_name);

if CACHE_FILE == 1 && fileexists(cache_file_stripped)
  newmodels = load(cache_file_stripped);
  newmodels = newmodels.models;
  return;
end


if CACHE_FILE == 1 && fileexists(cache_file)
  newmodels = load(cache_file);
  newmodels = newmodels.models;
  return;
end

DUMPDIR = sprintf('%s/www/svs/%s/',params.dataset_params.localdir, ...
                  new_models_name);

%display of SV pdfs disabled
%if CACHE_FILE==1 && params.dataset_params.display ==1 && ~exist(DUMPDIR,'dir')
%  mkdir(DUMPDIR);
%end

final_directory = ...
    sprintf('%s/models/%s/',params.dataset_params.localdir,...
            new_models_name);

%make results directory if needed
if CACHE_FILE == 1 && ~exist(final_directory,'dir')
  mkdir(final_directory);
end

% randomize chunk orderings
if CACHE_FILE == 1
  myRandomize;
  ordering = randperm(length(models));
else
  ordering = 1:length(models);
end

models = models(ordering);
allfiles = cell(length(models), 1);
for i = 1:length(models)
  filer = '';
  m = models{i};
  
  [complete_file] = sprintf('%s/%s.mat',final_directory,m.name);
  [basedir, basename, ext] = fileparts(complete_file);
  filer2fill = sprintf('%s/%%s.%s.mat',basedir,basename);
  filer2final = sprintf('%s/%s.mat',basedir,basename);  
  
  allfiles{i} = filer2final;
  
  % Check if we are ready for an update
  filerlock = [filer2final '.mining.lock'];

  if CACHE_FILE == 1
    if fileexists(filer2final) || (mymkdir_dist(filerlock) == 0)
      continue
    end
  end
  
  % Add training set and training set's mining queue 
  m.train_set = train_set;
  m.mining_queue = esvm_initialize_mining_queue(m.train_set);
  
  % Add mining_params, and params.dataset_params to this exemplar
  m.mining_params = params;
  m.dataset_params = params.dataset_params;

  % Append '-svm' to the mode to create the models name
  m.models_name = new_models_name;
  m.iteration = 1;
  
  %if we are a distance function, initialize to uniform weights
  if isfield(params,'wtype') && ...
        strcmp(params.wtype,'dfun')==1
    m.model.w = m.model.w*0-1;
    m.model.b = -1000;

  end

  % The mining queue is the ordering in which we process new images  
  keep_going = 1;

  while keep_going == 1
  
    %Get the name of the next chunk file to write
    filer2 = sprintf(filer2fill,num2str(m.iteration));

    if ~isfield(m,'mining_stats')
      total_mines = 0;
    else
      total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
    end
    m.total_mines = total_mines;
    m = esvm_mine_train_iteration(m, params.training_function);

    if ((total_mines >= params.train_max_mined_images) || ...
          (isempty(m.mining_queue))) || ...
          (m.iteration == params.train_max_mine_iterations)

      keep_going = 0;      
      %bump up filename to final file
      filer2 = filer2final;
    end
    
    %HACK: remove train_set which causes save issue when it is a
    %cell array of function pointers
    msave = m;
    m = rmfield(m,'train_set');
    
    %Save the current result
    if CACHE_FILE == 1
      savem(filer2,m);
    else
      allfiles{i} = m;
    end
    m = msave;
    
    % if params.dataset_params.display == 1

    %   if params.dataset_params.write_after_display == 1
    %     exid = ordering(i);
    %     filer = sprintf('%s/%s.%s.%05d.png', DUMPDIR, 'train', ...
    %                     m.cls,exid);
        
    %     if fileexists(filer)
    %       continue
    %     end
    %   end
      
    %   figure(445);
    %   clf;
    %   showI = esvm_show_det_stack(m,5,5);
    %   imagesc(showI);
    %   title('Exemplar and Top Dets');
    %   drawnow;
      
    %   figure(235)
      
    %   rpos = m.model.w(:)'*m.model.x-m.model.b;
    %   rneg = m.model.w(:)'*m.model.svxs - m.model.b;
    %   clf;
    %   plot(sort(rpos,'descend'),'r.');
    %   hold on;
    %   plot(length(rpos)+(1:length(rneg)),rneg,'b.');
    %   drawnow;

    %   if params.dataset_params.write_after_display == 1
    %     set(gcf,'PaperPosition',[0 0 20 20]);
    %     imwrite(showI,filer);
    %   end
    % end
    
    %delete old files
    if m.iteration > 1
      for q = 1:m.iteration-1
        filer2old = sprintf(filer2fill,num2str(q));
        if fileexists(filer2old)
          if CACHE_FILE == 1
            delete(filer2old);
          end
        end
      end
    end
    
    if keep_going==0
      fprintf(1,' ### End of training... \n');
      break;
    end
    
    m.iteration = m.iteration + 1;
  end %iteratiion
  
  try
    if CACHE_FILE == 1
      rmdir(filerlock);
    end
  catch
    fprintf(1,'Cannot delete %s\n',filerlock);
  end
end

if CACHE_FILE == 0
  newmodels = allfiles;
  return;
end

[allfiles] = sort(allfiles);

%Load all of the initialized exemplars
CACHE_FILE = 1;
STRIP_FILE = 1;

if new_models_name(1) == '-'
  CACHE_FILE = 0;
  STRIP_FILE = 0;
end

DELETE_INITIAL = 0;
newmodels = esvm_load_models(params.dataset_params, new_models_name, allfiles, ...
                          CACHE_FILE, STRIP_FILE, DELETE_INITIAL);


function savem(filer2,m)
save(filer2,'m');
