function [ X_folds, Y_folds ] = kfoldCV( X, Y, k )

  N = size(X,1);
  
  % permutacao dos dados
  rp = randperm(N); % permuta os indices
  X = X(rp,:); % aplica permutacao em X
  Y = Y(rp,:); % aplica permutacao em Y

  tamFold = floor(N/k); %Define a qtd de dados em cada fold

  X_folds = cell(k, 1); %Cria estruturas para os folds
  Y_folds = cell(k, 1);
  
  for fold = 1:k 
    %Calcula intervalo de indice dos dados que formarao o fold
    ini = ((fold-1)*tamFold)+1;
    fim = ini+tamFold-1;
    
    X_folds{fold} = X(ini:fim,:); %Define os dados para o fold
    Y_folds{fold} = Y(ini:fim,:);
  end
  

end

