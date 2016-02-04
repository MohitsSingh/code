function [successors,predecessors,current] = PSLz2a_get_relatives(points,imsize);
% [successors,predecessors,current] = PSLz3_get_neighbors(points,imsize)
%
% Goes to each skeleton point, finds neighbors in potential successor/predecessor locations
% and gets their corresponding features (orient, ener and whether they are `alive')
% Follows the paper by Nevatia & Babu, and uses the kdtree code by G. Schechter for
% efficient nearest neighbor search. 
% 
% Iasonas Kokkinos <jkokkin@stat.ucla.edu>
% 10/10/2007


scind   = points.scind;
indexes = points.indexes;
scl     = points.scl;
orient  = discretize_orientation(points.theta+pi/2,4);
ener    = points.ener;

[r_m,r_n] = ind2sub(imsize,indexes);
dm = 2;
descriptor_points = [r_m,r_n,scind/20]; % ,0*ener,cos(2*angl),sin(2*angl)];
TreeRoot_idx = kdtree(single(descriptor_points));


%% potential neighbor locations
%% NOTE: U corresponds to decreasing the matrix index.
% UL  U  UR
% L      R
% DL  D  DR

UL = [-1;-1]; U  = [-1; 0]; UR = [-1; 1];
L  = [0;-1];                R  = [0; 1];
DL = [1;-1];  D  = [1;0];   DR = [1; 1];

%% Depending on the orientation of the curve,
%% a subset of neighbors is considered.
%% These are the  considered scenaria:
%% the local orientation is one of
%% 1 : -   2:\   3: |   4: /
%% and the possible successor/predecessor neighbors

%% construct neighbor indexes, corresponding to
%% each line orientation scenario
%% e.g.: the successor of a horizontal line -neigh_sc(1,:)-can be
%% (a) the Up-Right neighbor, (b) the Right neighbor (c) the Down Right neighbor

neighs_sc(1,:,:) = [UR,R,DR]; neighs_pr(1,:,:) = [UL,L,DL];
neighs_sc(2,:,:) = [R,DR,D];  neighs_pr(2,:,:) = [L,UL,U];
neighs_sc(3,:,:) = [DL,D,DR]; neighs_pr(3,:,:) = [UL,U,UR];
neighs_sc(4,:,:) = [L,DL,D] ; neighs_pr(4,:,:) = [U,UR,R];

% discretize orientation of line element

for it = 1:2,
    if it ==1,
        neighs_used = neighs_sc;
    else
        neighs_used = neighs_pr;
    end

    %% loop over the three possible neighbors:
    for n_ind = 1:3,

        %% batch mode code below yields
        %% c_m(i)= neigs_sc(types(i),1,n_ind);
        %% c_n(i)= neigs_sc(types(i),2,n_ind);

        idx_m = orient + (1-1)*4+  4*2*(n_ind-1);
        idx_n = orient + (2-1)*4+  4*2*(n_ind-1);

        c_m = neighs_used(idx_m) + r_m;
        c_n = neighs_used(idx_n) + r_n;
        coord_h = c_m + (c_n-1)*imsize(1);
        
        %% for each of the three neighbors,
        %% find if the neighbor is actually `alive':
        
        %% 1) find closest point to the location of the potential neighbor
        pt_sent  =  [c_m,c_n,scind/20];
        
        [idx,pout] = kdtree_closestpoint(TreeRoot_idx,single(pt_sent));
        n_dist     = sqrt(sum(pow_2(pout  - pt_sent),2));
            
        scales_neigh = scl(idx);
        idx_m_n =  r_m(idx); idx_n_n = r_n(idx);
        
        %% require that the point found is exactly on the same location
        %% and not too far away in scale
        dist_location = abs(c_m - idx_m_n)  + abs(c_n - idx_n_n);
        dist_scale    = log(scales_neigh./scl);
        neigh_active(n_ind,:) = (dist_location<1)&(abs(dist_scale)<.5);
        % get its (quantized) orientation
        neigh_orient(n_ind,:) = orient(idx)';
        % and the strength of the ridge indicator function
        neigh_ener(n_ind,:) = ener(idx)';
        neigh_ind(n_ind,:)  = indexes(idx)';
        neigh_dist(n_ind,:) = n_dist';
    end
    
    fields_wt = {'neigh_active','neigh_orient','neigh_ener','neigh_ind','neigh_dist'};
    compress_structure;
    if it==1,
        successors = structure;
    else
        predecessors = structure;
    end
end

clear functions;
fields_wt = {'orient','ener','indexes','scl'}; 
compress_structure; current = structure;


function res = discretize_orientation(continuous,ndiscr);
% res = discretize_orientation(continuous,ndiscr)
%
% quantizes a continuous orientation angle into ndiscr values

norm = mod(continuous,pi)/pi;
step = 1/(2*ndiscr);
rotated = norm + step;
discr = floor(rotated*ndiscr);
res = mod(discr+1,ndiscr);
res = res + ndiscr*(res==0);