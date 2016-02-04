function neighbors = calcAdjancency( Mask )
%Use: Each on-zero element will have row with adjacent pixels (4-conn)
%     marked as 1
%Input:
%   Mask - Binary mask for building the adjacency matrix
%Output:
%   neighbors - An NxN Adjacency Matrix of all elements in the mask

NUM_OF_NEIGHBORS = 4;%defining a constant

[height, width]      = size(Mask);

%row_mask - row indexes of elements in a ROI/Mask
%col_mask - col indexes of elements in a ROI/Mask
[row_mask, col_mask] = find(Mask);

neighbors = zeros(length(row_mask), length(row_mask));

% Convert to indexes
roi_idxs = sub2ind([height, width], row_mask, col_mask);

for k = 1:size(row_mask, 1),
    %4 connected neighborhood
    connected_4 = [row_mask(k), col_mask(k)-1;%left
                   row_mask(k), col_mask(k)+1;%right
                   row_mask(k)-1, col_mask(k);%top
                   row_mask(k)+1, col_mask(k)];%bottom

    ind_neighbors = sub2ind([height, width], connected_4(:, 1), connected_4(:, 2));
    
    for neighbor_idx = 1:NUM_OF_NEIGHBORS,
        adjacent_pixel_idx = binaraysearchasc(roi_idxs, ind_neighbors(neighbor_idx));
        neighbors(k, adjacent_pixel_idx) = 1;
    end
    
end


end

% % Binary search. 
% % Search 'sval' in sorted vector 'x', returns index of 'sval' in 'x'
% %  
% % INPUT:
% % x: vector of numeric values, x should already be sorted in ascending order
% %    (e.g. 2,7,20,...120)
% % sval: numeric value to be search in x
% %  
% % OUTPUT:
% % index: index of sval with respect to x. If sval is not found in x
% %        then index is empty.
% % % --------------------------------
% % % Author: Dr. Murtaza Khan
% % % Email : drkhanmurtaza@gmail.com
% % % --------------------------------
%Dowloaded from matlab central:
%http://www.mathworks.com/matlabcentral/fileexchange/11287-binary-search-for-numeric-vector/content/binarysearchnumvec/binaraysearchasc.m

function index = binaraysearchasc(x,sval)

index=[];
from=1;
to=length(x);

while from<=to
    mid = round((from + to)/2);    
    diff = x(mid)-sval;
    if diff==0
        index=mid;
        return
    elseif diff<0   % x(mid) < sval
        from=mid+1;
    else              % x(mid) > sval
        to=mid-1;			
    end
end
end
