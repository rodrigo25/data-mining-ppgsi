function [ Xtr, Ytr, Xt, Yt ] = holdout( X, Y, perc )
  
  N = size(X,1);

  % permutacao dos dados
  rp = randperm(N); % permuta os indices
  X = X(rp,:); % aplica permutacao em X
  Y = Y(rp,:); % aplica permutacao em Y

  tamTr = round(perc*N); % qtd exemplos de uma classe para treinamento
  
  Xtr = X(1:tamTr,:); %Separa o conj de treinamento
  Ytr = Y(1:tamTr,:);
  
  Xt  = X(tamTr+1:end,:); %Separa o conj de treinamento
  Yt  = Y(tamTr+1:end,:);
  
end

