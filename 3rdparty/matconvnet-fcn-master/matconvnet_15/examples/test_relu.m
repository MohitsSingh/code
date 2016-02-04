X = gpuArray(zeros(1000,1000,20));
% X = zeros(100,100,20);
N = 1000;
%%
disp('using 0')
tic
for i = 1:N
    y = max(X,0);
end
toc

%%
disp('using single(0)')
tic
my_z = single(0);
for i = 1:N
    y = max(X,single(0));
end
toc
%%
disp('using single(0) precomputed')
tic
my_z = single(0);
for i = 1:N
    y = max(X,my_z);
end
toc

%%
disp('using gpuArray(0)')
tic
for i = 1:N
    y = max(X,gpuArray(0));
end
toc


%%
disp('using gpuArray(0) precomputed')
tic
my_z = gpuArray(single(0));
for i = 1:N
    y = max(X,my_z);
end
toc

