[train_label, train_feature] = libsvmread('train.libsvm');
model = svmtrain(train_label, train_feature, '-t 0 -c 0.001');
[predict_label, accuracy, dec_values] = svmpredict(train_label, train_feature, model);
svm = model.sv_indices;
feature = train_feature(svm,:);
label = train_label(svm);
%w*x + b = 0
%libsvm_a is the lagrangian multipliers
libsvm_a = model.sv_coef;
%libsvm_w is w in primal and dual
libsvm_w = sum(diag(libsvm_a)*feature)';