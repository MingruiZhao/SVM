load('train.mat')
%change all 0 to 1
for i = 1:8500
    if y(i) == 0
        y(i) = -1;
    end
end
h = (y'*y).*(X*X');
cvx_begin
%a is lagrangian multipliers
%w is weight vector which has 200 dimensions
%b is scaler
    variables a(8500)
    maximize (sum(a) - 1/2 * ((a.') * h * a));
    subject to
    %constraints
        y*a == 0;
        a >= 0;
        a <= 0.001;
cvx_end
%equations to reconstruct w and b in primal
w = (a.*y')'*X;
b = mean(y) - sum(a'.*y*X*X')/8500;
train_y = X*w.' + b;
for i = 1:8500
    if train_y(i) < 0
        train_y(i) = -1;
    end
end

for i = 1:8500
    if train_y(i) > 0
        train_y(i) = 1;
    end
end
sum(train_y == y')/8500;
load('test.mat')
%this part is used to test our accuracy by using test.mat
test_y = X*w.' + b;
for i = 1:1500
    if test_y(i) < 0
        test_y(i) = 0;
    end
end

for i = 1:1500
    if test_y(i) > 0
        test_y(i) = 1;
    end
end
sum(test_y == y')/1500