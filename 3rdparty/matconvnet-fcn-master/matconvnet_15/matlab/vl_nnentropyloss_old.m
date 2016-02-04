function Y = vl_nnentropyloss(X,c,dzdy)
%VL_NNENTROPYLOSS CNN entropy loss to encourage high information content.

% Amir Rosenfeld, Jan 2016.

%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

% we don't care at all about class labels for now, this is left for later. 
% if numel(c) == sz(4)
%   % one label per image
%   c = reshape(c, [1 1 1 sz(4)]) ;
% end
% if size(c,1) == 1 & size(c,2) == 1
%   c = repmat(c, [sz(1) sz(2)]) ;
% end
% 
% % one label per spatial location
% sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
% assert(isequal(sz_, [sz(1) sz(2) sz_(3) sz(4)])) ;
% assert(sz_(3)==1 | sz_(3)==2) ;
% 
% % class c = 0 skips a spatial location
% mass = single(c(:,:,1,:) > 0) ;
% if sz_(3) == 2
%   % the second channel of c (if present) is used as weights
%   mass = mass .* c(:,:,2,:) ;
%   c(:,:,2,:) = [] ;
% end
% 
% % convert to indexes
% c = c - 1 ;
% c_ = 0:numel(c)-1 ;
% c_ = 1 + ...
%   mod(c_, sz(1)*sz(2)) + ...
%   (sz(1)*sz(2)) * max(c(:), 0)' + ...
%   (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% compute softmaxloss

% X = X+randn(size(X)); % A noisy loss function !! :-)

addDiversity = true;

% Xmax = max(X,[],3) ;
% ex = exp(bsxfun(@minus, X, Xmax));
% Y = bsxfun(@rdivide, ex, sum(ex,3)) ; % Y has now column-normalized probabilities (over classes);
Y = X;
%n = sz(1)*sz(2) ;
if nargin <= 2    
  Y = -sum(Y(:).*log(Y(:)));
  z_row = sum(Y,4);
  q_row = bsxfun(@rdivide, Y,z_row); % G_ij
  Y = Y -sum(q_row(:).*log(q_row(:)));
%   Y
else
  d_diversity = zeros(size(Y),'single');
  if addDiversity
%       Xmax_row = max(X,[],4) ;
%       ex_row = exp(bsxfun(@minus, X, Xmax_row)); % X_ij : row-normalized probabilites (over batch).
      % normalize the rows...
      
      opts = 1;
      if opts==1
          z_row = gather(sum(Y,4));
          YY = gather(Y);
          q_row = bsxfun(@rdivide, Y,z_row); % G_ij
          dlogs = -gather(1+log(q_row));
          % local derivatives...
          
          %d_q = -bsxfun(@rdivide,q_row,(z_row.^2));          
          for i = 1:size(Y,3)
              for j = 1:size(Y,4)
                  for l_ = 1:size(YY,4)
                      if l_==j
                          M = z_row(i)-YY(1,1,i,j);
                      else
                          M = -YY(1,1,i,j);
                      end
                      d_diversity(1,1,i,j) = d_diversity(1,1,i,j) - dlogs(1,1,i,l_)*M;
                  end
              end
          end
          d_diversity = bsxfun(@rdivide,d_diversity,z_row.^2);
          
%           
%           M = bsxfun(@minus,z_row,Y);
%           d_diversity = bsxfun(@rdivide,M,(z_row).^2).*dlogs;
      else
          d_diversity = -2*Y;
      end
  end
  clf; imagesc(squeeze(Y));drawnow
%     pause
  Y = d_diversity;
  Y = -log(Y) -1 + d_diversity;
    
%   Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
%   Y(c_) = Y(c_) - 1;
%   Y = bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ;
end
