function [model] = grid_search(train_labels,train_data)
%GRID_SEARCH Summary of this function goes here
%   Detailed explanation goes here
v_crossvalidation = 2;
C = 2.^(-3:2:15);
gamma = 2.^(-15:2:3);

kernel_type = 2; % rbf

[C_,gamma_]=  meshgrid(C,gamma);
accuracies = zeros(size(C_));
n = 0;
for iC = 1:length(C)
    for iGamma = 1:length(gamma)
        n = n+1;
        libsvm_options = sprintf('-s 0 -t %d -c %f -g %f -v %d',...
            kernel_type, C(iC), gamma(iGamma), v_crossvalidation);
        model = svmtrain(train_labels, train_data, libsvm_options);
        %         accuracies(n) = model;
        accuracies(iGamma,iC) = model;
    end
end

figure,contour(C_,gamma_,accuracies);
set(gca,'XScale','log','YScale','log');
xlabel('C');
ylabel('gamma');
colorbar;

[m,im] = max(accuracies(:));
best_c = C_(im);
best_gamma = gamma_(im);

libsvm_options = sprintf('-s 0 -t %d -c %f -g %f',...
    kernel_type,best_c,best_gamma);

model = svmtrain(train_labels, train_data, libsvm_options);


end

