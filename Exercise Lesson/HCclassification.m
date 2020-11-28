function [weight, stats] = HCclassification(TrainX, TrainY, threshold, alpha, sdflag, muflag)
% The function HCclassification gives a prediction of label for testing
% data, with the information from train data, according to the procedure in
% Donoho and Jin (2008).
% 
% Function: [weight, stats] = HCclassification(TrainX, TrainY, threshold, alpha, sdflag, muflag)
% 
% Inputs:
%   TrainX   N-by-P matrix of predictors for train data set with one row 
%            per observation and one column per predictor.
%   TrainY   N-by-1 matrix of class labels "1" or "0"
%   threshold Choice of thresholding functions, can be 'clip', 'soft', or
%             'hard'
%            default 'hard'
%   alpha    The proportion of features with small p-values to calculate HC
%            threshold. 
%            default  0.2
%   sdflag   optional type of proxy sd: 
%              0 - std; 2 - floored at median; 1 - add median ;
%            default  1
%   muflag   optional type of proxy mu: 0, std, 1, average two means
%            default 1
% Outputs
%   weight   p-by-1 vector, showing the weight for each feature
%   stats    4-by-1 struct, including the normalization parameters for test
%            data and the statistic
%      stats.xbar   p-by-1 chosen mu proxy
%      stats.s      p-by-1 chosen standard deviation proxy
%      stats.HCT    chosen threshold
%      stats.HC     p-by-1 HC score for each feature
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
if (nargin<1 || isempty(TrainX))
  error 'Please input the train data'
end

if (nargin<2||isempty(TrainY))
    error 'Please input the label for train data'
end

if (nargin<3||isempty(threshold))
    threshold = 'hard';
end

if (nargin<4||isempty(alpha))
    alpha = 0.2;
end

if (nargin<5||isempty(sdflag))
    sdflag = 1;
end

if (nargin<6||isempty(muflag))
    muflag = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%  Main Function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[p, n] = size(TrainX);
if( n ~= length(TrainY)),
    error('WRONG SIZE: TrainX and TrainY arrays do not fit')
end

%Divide the data into 2 groups with different labels
Train0 = TrainX(:, TrainY == 0);
Train1 = TrainX(:, TrainY == 1);

%Calculate Z-score
 mu1   = mean(Train0, 2);  mu2    = mean(Train1, 2); %Mean for two groups
 if(muflag == 1),
    xbar     = (mu1+mu2)/2;
elseif(muflag == 2 ),
    xbar     =  mean(TrainX, 2);
else
    error('Improperly specified muflag, valid values are 0,1')
end

 s1    = std(Train0, 0, 2);   s2    = std(Train1, 0, 2); %SD for two groups
 n1    = size(Train0, 2);  n2    = size(Train1, 2);
 s     = (n1-1)*s1.^2 + (n2-1)*s2.^2;
 s     = sqrt(s/(n-2));
 smed  = median(s);

 if(sdflag == 1)
     s     = smed + s;
 elseif(sdflag == 2)
     s     = max(smed,s);
 elseif(sdflag == 0)
     s     = s;
 else
     error('Improperly specified sdflag, valid values are 0,1,2')
 end
 
 zscore     = (mu1 - mu2)./s/sqrt(1/n1 + 1/n2);  
% zscore     = (zscore - mean(zscore))/std(zscore);

 %Find the HC threshold with Z-score
pval  = 2.*(1 - normcdf(abs(zscore)));
kk    = (1:p)'/(1 + p);
psort = sort(pval);
HC  = (kk - psort)./sqrt(kk - kk.^2);
HC  = HC(1:round(alpha*p));
L   = find(HC == max(HC));
ind1 = find(pval <= psort(L) & zscore >0);
ind2 = find(pval <= psort(L) & zscore <= 0);
thr_inx = pval == psort(L);
absz    = abs(zscore);
HCT     = max(absz(thr_inx));

% Calculate the weights
weight = 0*zscore; 
if strcmp(threshold,  'clip')
    weight(ind1) = 1; weight(ind2) =-1; 
    % clip thresholding
elseif strcmp(threshold, 'soft')
    weight(ind1) =  abs(zscore(ind1))-HCT;
    weight(ind2) = -(abs(zscore(ind2))-HCT);
    % soft thresholding
elseif strcmp(threshold,'hard')
    weight(ind1) =  abs(zscore(ind1));
    weight(ind2) = -abs(zscore(ind2));
    % hard thresholding
else
    error 'The choice of thresholding function must be 'clip', 'soft', or 'hard'.'
end
    
% Record the statistics for every feature
stats.HC = HC; stats.HCT = HCT;  stats.xbar = xbar; stats.s = s;

end