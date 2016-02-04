%% General initialization
width=300;
height=width;
model_name='example.osg';
%% Basic examples
% Performs the rendering. However, since no output parameters were passed nothing is returned.
renderer(width,height,model_name);

% Performs the rendering and writes data files (depth.pgm and rendered.png). Note that this is _not_ the preferred way of usage.
renderer(width,height,model_name,1);

% Returns the depth matrix and rendered image to MATLAB. No output files are written. This is the preferred usage.
[depth, rendered]=renderer(width,height, model_name);

% Displays the results.
figure, imshow(depth); figure, imshow(rendered);

% Also returns the camera matrices - A,R and T as well as the unprojection matrix.
% Now 'unproject(125,149,1:3)' returns the world XYZ coordinate of the image point (x=148,y=124)
[depth, rendered, unproject, A, R, T]=renderer(width,height, model_name);

% Renders the mesh with a distance of 0.5, an elevation of 10 degrees, azimuth of 20 degrees and yaw of 30 degrees.
[depth, rendered, unproject, A, R, T]=renderer(width,height, model_name,0,0,0.5,10,20,30,'zxy');

% Providing the camera matrices returned from the previous example will yield the same results.
[depth, rendered, unproject]=renderer(width,height, model_name,0,0,A,R,T);
figure, imshow(rendered);

% View the 3D points of the model
figure, plot3(unproject(:,:,1),unproject(:,:,2),unproject(:,:,3),'.');
%% Pose estimation sanity
[depth, rendered, unproject, A, R, T]=renderer(width, height, model_name);
figure, imshow(rendered);

% collect dummy landmarks from the 2D rendered image and the 3D model points
% If you are fitting a different 2D image (texture), first resize it. e.g.
%   texture=imresize(your_texture,[height width]);
% Next, annotate m points (landmarks) from 'texture' (2D points) into pts_2D (m by 2 matrix, each row is XY) 
% and 'unproject' (3D points) into pts_3D (m by 3 matrix, each row is XYZ).

% pts_2D=[183,22; 135,106; 159,83; 132,187; 141,210; 130,229];
pts_2D=[151   171;   125   154;   183   191;   129   123;   141   126;   136   138];
hold on; plot(pts_2D(:,1),pts_2D(:,2),'r+','MarkerSize',7)
pts_3D=zeros(size(pts_2D,1),3);
for j=1:size(pts_2D,1)
    pts_3D(j,:)=unproject(pts_2D(j,2),pts_2D(j,1),1:3);
end;

% calibrate - estimate the new camera matrices
addpath('../calib/')
[A,R_new,T_new]=doCalib(width,height,pts_2D,pts_3D,A,[],[]);

% rendered_new should be almost equal to the original rendered
[~, rendered_new]=renderer(width, height, model_name,0,0,A,R_new,T_new);
figure, imshow(rendered_new);
%% Roate the model. First approach
for yaw=0:10:360
    [~, rendered]=renderer(width,height, model_name,0,0,0.5,10,20,yaw,'zxy');
    figure(1); imshow(rendered); pause;
end
%% Roate the model. Another approach
[~, rendered, ~, A, R, T]=renderer(width,height, model_name);
for yaw=deg2rad(0:10:360)
    dcm = angle2dcm(-yaw, 0, 0);
    [tmp, rendered]=renderer(width,height, model_name,0,0,A,dcm*R,T);
    figure(1); imshow(rendered); pause;
end
%% Re-render the mesh using a different image size
[~, rendered, ~, A, R, T]=renderer(width,height, model_name);
new_width=610; new_height=914;
A(1:2,3)=[new_width/2;new_height/2];
[~, rendered_new]=renderer(new_width, new_height, model_name,0,0,A,R,T);
imshow(rendered_new)
