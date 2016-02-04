function features = getDenseHOG(im, varargin)
% GETDENSEHOG   Extract dense HOG features
%   FEATURES = GETDENSEHOG(IM) extract dense HOG features from
%   image IM.

    opts.scales = logspace(log10(1), log10(.25), 5) ;
    opts.step = 2 ;
    opts.root = false ;
    opts.normalize = true ;
    opts.geometry = [4 4] ;
    opts.sigma = 0 ;
    opts.cellsize = 8;
    opts.edges_only = false;
    opts = vl_argparse(opts, varargin) ;

    dOpts = { 'step', opts.step, ...
                 'geometry', opts.geometry, ...
                 'cellsize', opts.cellsize, ...
                 'edges_only', opts.edges_only} ;

    if size(im,3)>1, im = rgb2gray(im) ; end
    im = im2single(im) ;
    im = vl_imsmooth(im, opts.sigma) ;

    for si = 1:numel(opts.scales)
      im_ = imresize(im, opts.scales(si)) ;

      [frames{si}, descrs{si}] = vl_dhog(im_, dOpts{:}) ;

      % root 
      if opts.root
        descrs{si} = sqrt(descrs{si}) ;
      end
      if opts.normalize
        descrs{si} = snorm(descrs{si}) ;
      end

      % store frames
      frames{si}(1:2,:) = (frames{si}(1:2,:)-1) / opts.scales(si) + 1 ;
      frames{si}(3,:) = opts.scales(si) ;
    end

    features.frame = cat(2, frames{:}) ;
    features.descr = cat(2, descrs{:}) ;
return;

function x = snorm(x)
    x = bsxfun(@times, x, 1./max(1e-5,sqrt(sum(x.^2,1)))) ;
return;

function [frames, descrs] = vl_dhog(im, varargin)
    
    opts.step = 2 ;
    opts.geometry = [4 4] ;
    opts.cellsize = 8;
    opts.edges_only = false;
    opts = vl_argparse(opts, varargin) ;
    
    edges_only = opts.edges_only;
    if (edges_only)
      [im_Gmag,im_Gdir] = imgradient(im);
      edge_orient = NaN(size(im));
      edge_mag = -1*ones(size(im));
      edges = edge(im,'canny',[],0.1);
      edges = imdilate(edges,ones(round(opts.cellsize/2)));
      if (sum(edges(:))>0)
        edge_orient(edges) = im_Gdir(edges); %values in [-180,180]
        edge_orient = edge_orient*pi/180; %values in [-180,180] -> [-pi,pi]
        edge_mag(edges) = im_Gmag(edges);
      end
      data = single(cat(3,edge_mag,edge_orient));
    end

    cellsize = opts.cellsize;
    d = size(vl_hog(single(zeros(cellsize+1)),cellsize),3);
    g_xy = opts.geometry;
    s = opts.step;
    if (mod(cellsize,s)~=0)
        error('bad step size');
    end
    nsteps = cellsize/s;
    
    imsize = size(im);
    r = floor((imsize(1)-g_xy(2)*cellsize+1)/s);
    c = floor((imsize(2)-g_xy(1)*cellsize+1)/s);
    descrs = zeros(g_xy(1)*g_xy(2)*d, r*c);
    frames = zeros(2, r*c);
    
    count = 0;
    for rs=1:nsteps
        for cs=1:nsteps
            r_start = (rs-1)*s+1;
            r_end = r_start+cellsize*floor((imsize(1)-r_start+1)/cellsize)-1;
            c_start = (cs-1)*s+1;
            c_end = c_start+cellsize*floor((imsize(2)-c_start+1)/cellsize)-1;
            if (edges_only)
                hog = vl_hog(data(r_start:r_end,c_start:c_end,:), cellsize, 'DirectedPolarField');%,'bilinearOrientations');%'UndirectedPolarField');
            else
                hog = vl_hog(im(r_start:r_end,c_start:c_end), cellsize);
            end

            rl = (size(hog,1)-g_xy(2)+1);
            cl = (size(hog,2)-g_xy(1)+1);
            
            for ri=1:rl
                for ci=1:cl
                    v = reshape(hog(ri:ri+g_xy(2)-1,ci:ci+g_xy(1)-1,:), [g_xy(1)*g_xy(2)*d,1]);
                    if (sum(v)==0)
                        continue;
                    end
                    count = count+1;
                    descrs(:,count) = v;
                    cp_start = cellsize*(ci-1)+c_start;
                    rp_start = cellsize*(ri-1)+r_start;
                    cp_end = cellsize*(ci+g_xy(1)-1)+c_start-1;
                    rp_end = cellsize*(ri+g_xy(2)-1)+r_start-1;
                    frames(:,count) = [(cp_start+cp_end)/2;(rp_start+rp_end)/2];
                end
            end
        end
    end
    
    frames(:,count+1:end)=[];%cleanup
    descrs(:,count+1:end)=[];%cleanup
    
    %sort
    pos = imsize(2)*frames(1,:)+frames(2,:);
    [~,I]=sort(pos);
    frames=frames(:,I);
    descrs=descrs(:,I);
    
return;