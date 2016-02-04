function X = vl_nnentropyloss(X,c,dzdy)
%VL_NNENTROPYLOSS CNN entropy loss to encourage high information content.

% Amir Rosenfeld, Jan 2016.

%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;


useRows = 1;
useCols = 1;

if nargin <= 2     % forward pass.
    %Y = -sum(Y(:).*log(Y(:)));
    
    P = bsxfun(@rdivide,X,sum(X,3));
    Q = bsxfun(@rdivide,X,sum(X,4));
    
    % normalize by rows.
    X = -useCols*sum(P(:).*log(P(:))) - ...
        useRows*sum(Q(:).*log(Q(:)));
    
    %   z_row = sum(Y,4);
    %   q_row = bsxfun(@rdivide, Y,z_row); % G_ij
    %   Y = Y -sum(q_row(:).*log(q_row(:)));
    %   Y
else
    %     d_diversity = zeros(size(Y),'single');
    
    X = squeeze(gather(X));
    sum_cols = sum(X,1);
    sum_rows = sum(X,2);
    P = bsxfun(@rdivide,X,sum_cols);
    Q = bsxfun(@rdivide,X,sum_rows);
    dP = zeros(size(X));
    dQ = zeros(size(X));
    L_p = log(P)+1;
    L_q = log(Q)+1;
    for i = 1:size(X,1)
        for j = 1:size(X,2)
            %             V = -X(i,j);
            % fill P : iterate over column elements
            for row = 1:size(X,1)
                v = -X(row,j);
                if row == i
                    v = v+sum_cols(j);
                end
                dP(i,j) = dP(i,j) - L_p(row,j)*v;
            end
            % fill Q : iterate over column elements
            for col = 1:size(X,2)
                v = - X(i,col);
                if col == j
                    v = v+sum_rows(i);
                end
                dQ(i,j) = dQ(i,j) - L_q(i,col)*v;
            end
        end
        
%         figure(1); clf;
%         subplot(2,2,1); imagesc(squeeze(X));
%         subplot(2,2,2); imagesc(squeeze(dP));
%         subplot(2,2,3); imagesc(squeeze(dQ));
%         drawnow;pause(.1)
    end
    
    dP = bsxfun(@rdivide,dP,sum_cols.^2);
    dQ = bsxfun(@rdivide,dQ,sum_rows.^2);
    
    figure(1); clf; 
%     subplot(2,2,1); 
    imagesc(squeeze(X)); drawnow; pause(.01)
%     subplot(2,2,2); imagesc(squeeze(dP)); 
%     subplot(2,2,3); imagesc(squeeze(dQ)); 
%     dpc(.1)
    X = useRows*dQ+useCols*dP;
    X = gpuArray(reshape(X,1,1,size(X,1),size(X,2)));
end
%     pause
% Y = d_diversity;
% Y = -log(Y) -1 + d_diversity;

%   Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
%   Y(c_) = Y(c_) - 1;
%   Y = bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ;
end
