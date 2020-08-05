load('train.mat')
%change all 0 to 1
for i = 1:8500
    if y(i) == 0
        y(i) = -1;
    end
end

cvx_begin 
%w is vector with 200 dimensions
%b is a scaler
%c is slack variable
    variables w(200) b c(8500)
    %let's set c = 0.001 to ensure the accuracy, when c is 0.001
    %the hyperplane will be smooth, so in testing phrase
    %the accuracy will be high
    minimize (0.5*w'*w + 0.001 * sum(c))
    subject to
    %contraints
      y'.*(X*w + b) >=  1 - c; 
      c>=0;
cvx_end
pop = cvx_optval;
train_y = X*w + b;
%because there are lots of data have the value between -1 and 1
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
sum(train_y == y')/8500
%this part is used to test our accuracy by using test.mat
load('test.mat')
test_y = X*w + b;
%because there are lots of data have the value between -1 and 1
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