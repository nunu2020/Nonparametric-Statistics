function [label, score] = HCclassification_fit(weight, xbar, s, Test)
% The function HCclassification gives a prediction of label for testing
% data, with the information from train data, according to the procedure in
% Donoho and Jin (2008).
% 
% Function: [label, score] = HCclassification_fit(weight, xbar, s, Test)
%
% Inputs:
%   weight   p-by-1 vector that shows the weight for each predictor
%   xbar     p-by-1 vector, estimated mean from HCclassification function
%   s        p-by-1 vector, estimated standard deviation from 
%            HCclassification function
%   Test     M-by-P matrix of predictors for data set to be predicted
%   
% Outputs
%   label    m-by-1 vector of estimated labels "1" or "0"
%   score    m-by-1 vector, showing classification score for test data
% 
%
% Example:
%  load('lungCancer.mat');
%  TrainData = lungCancertrain(:, 1:12533); TrainData = TrainData';
%  Test = lungCancer_test(1:149, 1:12533); Test = Test'; %The last two
%  observations do not have label information, and so they are excluded
%  [wt, stats] = HCclassification(TrainData, lungCancertrain(:, 12534), 'clip');
%  [label, score] = HCclassification_fit(wt, stats.xbar, stats.s, Test);
%  %Error Rate
%  sum(label ~= lungCancer_test(1:149,12534))
%
% Reference: 
% Donoho and Jin (2008) Higher criticism thresholding: Optimal feature
% selection when useful features are rare and weak


% Error checking
if (nargin<1 || isempty(weight))
  error 'Please input the classifier'
end

if (nargin<2||isempty(xbar))
    xbar = zeros(size(Test, 2), 1);
end

if (nargin<3||isempty(s))
    s = ones(size(Test, 2), 1);
end

if (nargin<4||isempty(Test))
    error 'Please input the test data'
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%  Main Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[p, m] = size(Test);
if( p ~= length(weight)|| p~=length(xbar) || p~=length(s))
    error('WRONG SIZE: Classifier and normalization parameter arrays do not fit')
end


% Normalize the test data
z = 0*Test;
for j = 1:m
    z(:,j) = (Test(:,j) - xbar)./s;
end

% Prediction
score = z'*weight;
label = (score <= 0)*1; 

end