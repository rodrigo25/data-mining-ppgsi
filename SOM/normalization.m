function [ DATA, mean_val, std_val, min_val, max_val ] = normalization( X, type )
% NORMALIZACAO DOS DADOS X

  [N, m] = size(X);

  mean_val = mean(X);
  std_val = std(X);
  min_val = min(X);
  max_val = max(X);
 
  if strcmp(type,'zscore') %z-score normalization
    DATA = (X-repmat(mean_val,N,1))./repmat(std_val,N,1);
  
  elseif strcmp(type,'minmax') % min-max normalization
    DATA = zeros(N,m);
    for i=1:m
      DATA(:,i) = (X(:,i)-min_val(i))/(max_val(i)-min_val(i));
    end
  end
  
end

