function [Ipos,Ineg] = my_esvm_generate_dataset(gt_data,false_images)  
% Generate a synthetic dataset of circles (see esvm_demo_train_synthetic.m)
% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm

% if nargin == 0
%   Npos = 3;
%   Nneg = 10
% elseif nargin == 1
%   Nneg = 10;
% end
% 
% A = zeros(39,39);
% A(20,20)=1;
% A = double(bwdist(A)<15);
% A = bwmorph(A,'remove');
% A = bwmorph(A,'dilate',2);
% Asave = repmat(A,[1 1 3]);

Ipos = cell(length(gt_data),1);
for i = 1:length(Ipos)
%   I = rand(100,100,3);
%   I = rand(50,50,3);
%   rscale = (rand*.8)+(1.0-.4);
%   A = imresize(Asave,rscale,'nearest');

%   I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
%   sub1 = ceil(rand.*(size(I,1)-size(A,1)-1));
%   sub2 = ceil(rand.*(size(I,2)-size(A,2)-1));
  %A2 = A + rand(size(A)).*(A<.9);
%   I2 = zeros(size(I));
%   I2(sub1+(1:size(A,1)),sub2+(1:size(A,2)),:)=A;

%   inds = find(I2);
%   I(inds) = 0;
  
%   Irand = rand(size(I));
%   I = .4*I+.6*Irand;
  
  %Ipos{i}.I = I;
  Ipos{i}.I = gt_data(i).sourceImage;
  
  recs.folder = '';
  recs.filename = gt_data(i).sourceImage;
  recs.source = '';
  iInfo = imfinfo(gt_data(i).sourceImage);
  %I = imread(gt_data(i));
  %[recs.size.width,recs.size.height,recs.size.depth] = size(I);
  [recs.size.width,recs.size.height,recs.size.depth] = deal(iInfo.Width,iInfo.Height,3); % TODO - assumes it's 3 channels
  recs.segmented = 0;
  recs.imgname = sprintf('%08d',i);
  recs.imgsize = [recs.size.width,recs.size.height,recs.size.depth];
  recs.database = '';

  object.class = gt_data(i).name;
  object.view = '';
  object.truncated = 0;
  object.occluded = 0;
  object.difficult = 0;
  object.label = gt_data(i).name;
  
  x = round(gt_data(i).polygon.x);
  y = round(gt_data(i).polygon.y);
  xmin = min(x); ymin = min(y);
  xmax = max(x); ymax = max(y);
  object.bbox = [xmin ymin xmax ymax];
  object.bndbox.xmin = object.bbox(1);
  object.bndbox.ymin = object.bbox(2);
  object.bndbox.xmax = object.bbox(3);
  object.bndbox.ymax = object.bbox(4);
  object.polygon = [];
  recs.objects = [object];
  %object.mask = [];
  %object.hasparts = 0;
  %object.par
  
  Ipos{i}.recs = recs;
  % Ipos{i}.bbox = 
  % Ipos{i}.cls = 'synthetic';
  % Ipos{i}.curid = sprintf('%05d',i);
  % filer = sprintf('%s.%d.%s.mat', Ipos{i}.curid, 1, ...
  %                 'synthetic');
  % Ipos{i}.filer = filer;
  % Ipos{i}.objectid = 1;
  if 0
  figure(1)
  clf
  imagesc(I)
  plot_bbox(bbs{i});

  pause
  end
end
Ineg = false_images;
% Ineg = cell(Nneg,1);
% for i = 1:Nneg
% %   I = rand(100,100,3);
% %   I = rand(50,50,3);
% %   I = max(0.0,min(1.0,imresize(I,[100 100],'bicubic')));
%   Ineg{i} = I;
% end