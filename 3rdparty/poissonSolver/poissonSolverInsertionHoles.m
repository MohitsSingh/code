function [imNew] = poissonSolver(imSrc, imDest, boxSrc, posDest)
%
% rectangular solver
% 
% parameters
% imSrc - source image
% imDest - destination image
% boxSrc - the rectangular box for the selection in source, [x0 x1 y0 y1]
% posDest - the upper left corner of the source should go to this position
% in the destination image [x0 y0]

% --------------------------------------------
% global variables
% --------------------------------------------

% laplacian = [0 1 0; 1 -4 1; 0 1 0];

% height and width of both the source image and the destination image
[heightSrc widthSrc] = size(imSrc);
[heightDest widthDest] = size(imDest);

% the four corners of the selected region in the source image
x0s = boxSrc(1);
x1s = boxSrc(2);
y0s = boxSrc(3);
y1s = boxSrc(4);

% the starting pixel of the region in the destination image
x0d = posDest(1);
y0d = posDest(2);

% the height and width of the region
heightRegion = y1s - y0s + 1;
widthRegion = x1s - x0s + 1;

% --------------------------------------------

%---------------------------------------------
% check for boundary conditions
%---------------------------------------------
% here, we make sure that the boundary of the region
% does not coinside with the boundary of either the source
% image or the destination image
if x0s <= 1 | x1s >= widthSrc | y0s <= 1 | y1s >= heightSrc | x0d <= 1 | y0d <= 1
    fprintf('Error - cannot handle such boundary condition\n');
end
        
%---------------------------------------------
% sparse matrix allocation
%---------------------------------------------
J = heightRegion;
L = widthRegion;
n = J*L;
fprintf('Matrix dimension = %d x %d\n', n, n);
M = spalloc(n, n, 5*n);

% also the boundary condition
b = zeros(1, n);

%---------------------------------------------
% construct the matrix here
%---------------------------------------------

tic
% imLaplacian = conv2(imSrc, -laplacian, 'same');

% matrix row count
count = 1;
for y = 1:heightRegion
    for x = 1:widthRegion

        % index of the vector
        i = (y-1)*L + x;        
         
        %------------------------------------------------------
        % if Neighbourhood(p) is in the interia of the region
        %------------------------------------------------------
        
        
        % gathering the coefficient for the matrix
        %------------------------------------------------------
        % if on the top
        if y ~= 1
            M(count, i-L) = -1;
        else % at the top boundary
            b(count) = b(count) + imDest(y+y0d-1, x+x0d);
        end
        
        % if on the bottom
        if y ~= heightRegion
            M(count, i+L) = -1;
        else
            b(count) = b(count) + imDest(y+y0d+1, x+x0d);
        end
        
        % if on the left
        if x ~= 1
            M(count, i-1) = -1;
        else
            b(count) = b(count) + imDest(y+y0d, x+x0d-1);
        end
        
        if x ~= widthRegion
            M(count, i+1) = -1;
        else
            b(count) = b(count) + imDest(y+y0d, x+x0d+1);
        end
        
        M(count, i) = 4;
        
        % construct the guidance field

        xd = x + x0d;
        yd = y + y0d;
        
        fpq1 = imDest(yd, xd) - imDest(yd-1, xd);
        fpq2 = imDest(yd, xd) - imDest(yd+1, xd);
        fpq3 = imDest(yd, xd) - imDest(yd, xd-1);
        fpq4 = imDest(yd, xd) - imDest(yd, xd+1);

        xv = x + x0s;
        yv = y + y0s;        
        
        gpq1 = imSrc(yv, xv) - imSrc(yv-1, xv);
        gpq2 = imSrc(yv, xv) - imSrc(yv+1, xv);
        gpq3 = imSrc(yv, xv) - imSrc(yv, xv-1);
        gpq4 = imSrc(yv, xv) - imSrc(yv, xv+1);
    
        v = 0;
        
        if abs(fpq1) > abs(gpq1)
            v = v + fpq1;
        else
            v = v + gpq1;
        end
        
        if abs(fpq2) > abs(gpq2)
            v = v + fpq2;
        else
            v = v + gpq2;
        end
        
        if abs(fpq3) > abs(gpq3)
            v = v + fpq3;
        else
            v = v + gpq3;
        end
        
        if abs(fpq4) > abs(gpq4)
            v = v + fpq4;
        else
            v = v + gpq4;
        end


        b(count) = b(count)+v;
        
        count = count + 1;   
    end
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
imRegion = reshape(x, widthRegion, heightRegion);


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
imNew = imDest;
imNew(y0d+1:y0d+heightRegion, x0d+1:x0d+widthRegion) = imRegion';


