function [ DATA, mean_val, std_val, min_val, max_val ] = normalization( X, type, mean_val, std_val, min_val, max_val )
% NORMALIZACAO DOS DADOS X

  [N, m] = size(X);

  %z-score normalization
  if strcmp(type,'zscore')
    if ~exist('mean_val','var')
      mean_val = mean(X);
    end
    if ~exist('std_val','var')
      std_val = std(X);
    end
    
    DATA = (X-repmat(mean_val,N,1))./repmat(std_val,N,1);
  
    min_val=[]; max_val=[];
    
  %min-max normalization
  elseif strcmp(type,'minmax') 
    
    if ~exist('min_val','var')
      min_val = min(X);
    end
    if ~exist('max_val','var')
      max_val = max(X);
    end
    
    DATA = zeros(N,m);
    for i=1:m
      DATA(:,i) = (X(:,i)-min_val(i))/(max_val(i)-min_val(i));
    end
    
    mean_val=[]; std_val=[];
    
  end
  
end

