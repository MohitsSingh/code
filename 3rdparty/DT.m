function [D R] = DT(img)
% Two-dimensional generalized distance transform
%
% Input: f - the sampled function
%
% Output: D - distance transform
%         R - power diagram
%
% Based on the paper:
% P. Felzenszwalb, D. Huttenlocher
% Distance Transforms of Sampled Functions
% Cornell Computing and Information Science Technical Report TR2004-1963,
% September 2004
%
%
% This is a simple MATLAB implmentation of the generalized distance
% transform algorithm. The function DT() gives the distance transform 
% of a 2D image by calling DT1() for each dimension. By using DT1(), 
% this could be easily extended to higher dimensions. It seems to have 
% problems with inf values, so for points in the image with "no" parabola 
% centered there, they should instead be given a large numeric value 
% (such as 1e10). I also modified the algorithm so that the second argument 
% returns the power diagram of the input. The power diagram is a diagram 
% where each point is assigned to the point that is closest to it with 
% respect to the distance transform. If all input points have the same 
% value, this function reduces to giving the standard distance transform 
% and the Voronoi diagram.
% 
% % EXAMPLE:
% % 
% % Create some random points
% X = randi(100,50,2);
% % Create an image with rando values at these points
% img = sparse(X(:,2), X(:,1), rand(50,1)*20,100,100);
% % Set all other values to a high number
% img(img==0) = 1e10;
% % Call the function
% [D R] = DT(img);
% % Plot the results
% figure;
% subplot(1,2,1);
% imagesc(D);
% title('Generalized Distance transform');
% axis image;
% subplot(1,2,2);
% imagesc(R);
% title('Power diagram');
% axis image;



D = zeros(size(img));
R = zeros(size(img));

for i = 1:size(img,1)
    [d r] = DT1(img(i,:));
    R(i,:) = r;
    D(i,:) = d;
end
for j = 1:size(img,2)
    [d r] = DT1(D(:,j));
    D(:,j) = d;
    R(:,j) = sub2ind(size(img), r, R(r,j));
end