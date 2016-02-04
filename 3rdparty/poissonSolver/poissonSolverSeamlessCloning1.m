function [imNew] = poissonSolverSeamlessCloning1(imSrc, imDest, imMask, offset)
%
% rectangular solver
% 
% parameters
% imSrc - source image
% imDest - destination image
% imMask - a black and white mask to indicate the irregular boundary
%
% posOffset - offset of corresponding pixel from the source to the
% destination
%

% --------------------------------------------
% global variables
% --------------------------------------------

laplacian = [0 1 0; 1 -4 1; 0 1 0];

% height and width of both the source image and the destination image
[heightSrc widthSrc] = size(imSrc);
[heightDest widthDest] = size(imDest);
[heightMask widthMask] = size(imMask);

% check if the mask is too big
if heightSrc < heightMask | widthSrc < widthMask | heightDest < heightMask | widthDest < widthMask
    fprintf('Error - the mask is too big\n');    
end
    
    
% the four corners of the selected region in the source image
% x0s = boxSrc(1);
% x1s = boxSrc(2);
% y0s = boxSrc(3);
% y1s = boxSrc(4);

% the offset between the source and the destination
xoff = offset(1);
yoff = offset(2);

% the height and width of the region
% heightRegion = y1s - y0s + 1;
% widthRegion = x1s - x0s + 1;



%---------------------------------------------
% check for boundary conditions
%---------------------------------------------
% here, we make sure that the boundary of the region
% does not coinside with the boundary of either the source
% image or the destination image
% if x0s <= 1 | x1s >= widthSrc | y0s <= 1 | y1s >= heightSrc | x0d <= 1 | y0d <= 1
%     fprintf('Error - cannot handle such boundary condition\n');
% end
        
% --------------------------------------------
% calculate the number of pixels that are 0
% for sparse matrix allocation
% --------------------------------------------
n = size(find(imMask), 1);

%---------------------------------------------
% sparse matrix allocation
%---------------------------------------------
% J = heightRegion;
% L = widthRegion;
% n = J*L;
fprintf('Matrix dimension = %d x %d\n', n, n);
M = spalloc(n, n, 5*n);

% also the boundary condition
b = zeros(1, n);

% temperary matrix index holder
% need to point the pixel position in the image to
% the row index of the solution vector to
imIndex = zeros(heightDest, widthDest);


fprintf('Building index\n');
tic

count = 0;
% now fill in the 
for y = 1:heightDest
    for x = 1:widthDest
        if imMask(y+yoff, x+xoff) ~= 0
            count = count + 1;            
            imIndex(y, x) = count;
        end
    end
end

toc
        
if count ~= n
    fprintf('Error - wrong matrix size\n');
end

        
        
%---------------------------------------------
% construct the matrix here
%---------------------------------------------

tic

% construct the laplacian image.
imLaplacian = conv2(imSrc, -laplacian, 'same');

% matrix row count
count = 0; % count is the row index
for y = 1:heightSrc
    for x = 1:widthSrc

        % if the mask is not zero, then add to the matrix
        if imMask(y, x) ~= 0

            % increase the counter
            count = count + 1;   
            
            % the corresponding position in the destination image
            yd = y - yoff;
            xd = x - xoff; 
             
            %------------------------------------------------------
            % if Neighbourhood(p) is in the interia of the region
            %------------------------------------------------------
            
            
            % gathering the coefficient for the matrix
            %------------------------------------------------------
            % if on the top
            if imMask(y-1, x) ~= 0
                % this pixel is already used
                % get the diagonal position of the pixel
                colIndex = imIndex(yd-1, xd);
                M(count, colIndex) = -1;
            else % at the top boundary
                b(count) = b(count) + imDest(yd-1, xd);
            end
            
            % if on the left
            if imMask(y, x-1) ~= 0
                colIndex = imIndex(yd, xd-1);
                M(count, colIndex) = -1;
            else % at the left boundary
                b(count) = b(count) + imDest(yd, xd-1);
            end            
            
            %------------------------------------------------------
            % now the harder case, since this is not allocated
            %------------------------------------------------------ 
            % if on the bottom            
            if imMask(y+1, x) ~= 0
                colIndex = imIndex(yd+1, xd);
                M(count, colIndex) = -1;
            else    % at the bottom boundary
                b(count) = b(count) + imDest(yd+1, xd);
            end
            
            % if on the right side
            if imMask(y, x+1) ~= 0
                colIndex = imIndex(yd, xd+1);
                M(count, colIndex) = -1;
            else    % at the right boundary
                b(count) = b(count) + imDest(yd, xd+1);
            end       
            
            M(count, count) = 4;
            
            % construct the guidance field	
            v = imLaplacian(y, x);
	
            b(count) = b(count)+v;

        end
    end
end

if count ~= n
    fprintf('Error - wrong matrix size\n');
end


toc

%---------------------------------------------
% solve for the sparse matrix
%---------------------------------------------
tic
% use bi-conjugate gradient method to solve the matrix
x = bicg(M, b', [], 300);
toc

% reshape x to become the pixel intensity of the region
% imRegion = reshape(x, widthRegion, heightRegion);

%---------------------------------------------
% now fill in the solved values
%---------------------------------------------
imNew = imDest;

fprintf('\nRetriving result, filling destination image\n');
tic
% now fill in the 
for y1 = 1:heightDest
    for x1 = 1:widthDest
        if imMask(y1+yoff, x1+xoff) ~= 0
            index = imIndex(y1, x1);
            imNew(y1, x1) = x(index);
        end
    end
end

toc


%---------------------------------------------
% show the region using in 3D
%---------------------------------------------


% x = x0:h1:x1;  y = y0:h2:y1;
% [xnew,ynew] = meshgrid(x,y);
% figure(50)
% mesh(xnew,ynew,unew,'EdgeColor','black');
% xlabel('x'); ylabel('y'); zlabel('Solution, u')
% 
% filename = '../images/poissonSolverResult.jpg';
% print( gcf, '-djpeg100', filename); 

%----------------------------------------------
% compose the image
%----------------------------------------------

% imNew(y0d+1:y0d+heightRegion, x0d+1:x0d+widthRegion) = imRegion';


