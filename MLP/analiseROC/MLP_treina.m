function [A, B, MSEtrain, MSEval, epochsExecuted] = MLP_treina(X, Yd, Xval, Ydval, h, mE, alfa0, alfaDecay, earlyStopping, V)
% MLPtreina treina uma MLP com 1 camada oculta para um dataset de N linhas,
% ne colunas e nc classes.
% X         - Training dataset, matriz (N,ne)
% Yd        - Training labels (N,nc)
% Xval      - Validation dataset
% Ydval     - Validation labels
% h         - Nº neuronios camada oculta
% mE        - max iterations
% alfa0     - Taxa de aprendizado inicial
% alfaDecay     -
%                   0: constante, sem decaimento
%                   1: decaimento pela metade step (a cada 5 epocas)
%                   2: dinamica, cai 10% se o erro do dataset de validacao
%                   sobe e sobe 10% caso contrario
% earlyStopping -
%                   0: nao para o treino se o MSE de validacao subir
%                   1: para o treino se o MSE de validacao subir
% V         - matriz (N,nc) de pesos para os erros, opcional
    
    [N, ne] = size(X);
    ns = size(Yd,2);
    
    if nargin < 10
        V = ones(N,ns);
        if nargin < 9
           earlyStopping = 0; 
        end
    end
    
    X = [ones(N,1),X]; % add bias

    A = rand(h, ne+1); 
    B = rand(ns, h+1); 

    maxMSE = 1e-5;

    MSEtrain = [];
    MSEval = [];
    alfa = alfa0;
    minAlfa = .001;
    maxAlfa = .9;
    epochsExecuted = 0;

    for it=1:mE
      
      epochsExecuted = it;
      
      %Feedfoward
      Zin = X*A';
      Z =  1./(1+exp(-Zin));
      Yin = [ones(N,1),Z]*B';
      Y = 1./(1+exp(-Yin));

      %Training error
      err = (Y-Yd);
      mse = sum(sum( err.^2 ))/N;
    
      %Validation error
      Yval = MLPsaida(Xval,A,B);
      errVal = (Yval-Ydval);
      mseVal = sum(sum( errVal.^2 ))/size(Xval,1);
     
      %Backpropagation
      gradB = (V.*err).*(1-Y).*Y;
      gradA = (gradB*B(:,2:end)).*(Z.*(1-Z));

      deltaA = alfa * gradA' * X;
      deltaB = alfa * gradB' * [ones(N,1),Z];

      A = A - deltaA;
      B = B - deltaB;
      
      if mse<maxMSE
          break;
      end
      
      if mod(it,20) == 0
         savedA = A;
         savedB = B;
      end
      
      % if early stopping is enabled
      % and if the last 20 epochs show no improvement
      if earlyStopping && it > 1 && mod(it-1,20) == 0 && mseVal >= MSEval(it-1)
          % check last epochs improvements
          % assume no improvement
          lastEpochsNoImprovement = 1;
          for i=19:-1:1
              if MSEval(it-20+i) < MSEval(it-20+i-1) 
                  % some improvement found
                  lastEpochsNoImprovement = 0;
                  break;
              end
          end
          if lastEpochsNoImprovement == 1
              % Restores previously saved weights
              A = savedA;
              B = savedB;
              % stops execution and returns
              break;
          end
      end
      
      %Learning rate update
      if alfaDecay == 0
          alfa = alfa0;
      elseif alfaDecay == 1 && mod(it,5) == 0 && alfa > minAlfa
          alfa = .5 * alfa;
      elseif alfaDecay == 2 && it > 1 
          if MSEval(it-1) < mseVal
              if alfa > minAlfa
                  alfa = alfa * .9;
              end
          elseif alfa < maxAlfa
              alfa = alfa * 1.1;
          end
      end

      %Stores some values for plotting
      MSEtrain = [MSEtrain;mse];
      MSEval = [MSEval;mseVal];
      
      %{
      figure(1)
      clf;
      plot(MSEtrain, 'b');
      hold on
      plot(MSEval, 'r');
      title(sprintf('EQM X Epocas (h=%d, mE=%d)',h,mE)); 
      xlabel('Epocas');
      ylabel('Erro Quadratico Medio');            
      legend('train', 'validation');
      %}
      %fprintf('train = %.8f\tval = %.8f\n',mse,mseVal);
      %fprintf('%.8f\n',alfa);
    end

    %close(figure(1));
end