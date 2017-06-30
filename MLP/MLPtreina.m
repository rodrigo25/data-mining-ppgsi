function [A, B, MSEtrain, MSEval, epochsExecuted] = MLPtreina(X, Yd, Xval, Ydval, L, h, mE, alfa0, alfaDecay, earlyStopping, V)
% MLPtreina treina uma MLP com 1 camada oculta para um dataset de N linhas,
% ne colunas e nc classes.
% X         - Training dataset, matriz (N,ne)
% Yd        - Training labels (N,nc)
% Xval      - Validation dataset
% Ydval     - Validation labels
% L         - Nº de camadas ocultas
% h         - Nº neuronios em cada camada oculta
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
    
    if nargin < 11
        V = ones(N,ns);
        if nargin < 10
           earlyStopping = 0; 
        end
    end
    
    X = [ones(N,1),X]; % add bias

    A = cell(L,1);
    A{1} = rand(h, ne+1);
    for i=2:L
       A{i} = rand(h/(2^(i-1)), h+1);
    end
    B = rand(ns, (h/(2^(L-1)))+1); 

    maxMSE = 1e-5;

    MSEtrain = [];
    MSEval = [];
    alfa = alfa0;
    minAlfa = .001;
    maxAlfa = .1;
    epochsExecuted = 0;

    Zin = cell(L,1);
    Z = cell(L,1);
    gradA = cell(L,1);
    deltaA = cell(L,1);
    
    alfas = [alfa];
    
    for it=1:mE
      
      epochsExecuted = it;
      
      %Feedfoward
      Zin{1} = X*A{1}';
      Z{1} =  1./(1+exp(-Zin{1}));
      for i=2:L
         Zin{i} = [ones(size(Z{i-1},1),1) Z{i-1}] * A{i}';
         Z{i} = 1./(1+exp(-Zin{i}));
      end
      Yin = [ones(N,1),Z{L}]*B';
      Y = 1./(1+exp(-Yin));

      %Training error
      err = (Y-Yd);
      mse = sum(sum( err.^2 ))/N;
    
      %Validation error
      Yval = MLPsaida(Xval,L,A,B);
      errVal = (Yval-Ydval);
      mseVal = sum(sum( errVal.^2 ))/size(Xval,1);
     
      %Backpropagation
      gradB = (V.*err).*(1-Y).*Y;
      g = gradB;
      G = B;
      for i=L:-1:1
         gradA{i} = (g*G(:,2:end)).*(Z{i}.*(1-Z{i})); 
         if i > 1
             Gin = [ones(size(Z{i-1},1),1), Z{i-1}];
         else
             Gin = X;
         end
         deltaA{i} = gradA{i}' * Gin;
         g = gradA{i};
         G = A{i};
      end

      deltaB = gradB' * [ones(N,1),Z{L}];

      for i=1:L
          A{i} = A{i} - alfa * deltaA{i};
      end
      B = B - alfa * deltaB;
      
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
      elseif alfaDecay == 2
          if it==1
             mseAux = +Inf;
          end
          if mseAux > mse
              if alfa > .005
                alfa = alfa*.9;
              end
          else
             alfa = alfa*1.1; 
          end
          mseAux = mse;
          %{
          Aaux = A;
          Baux = B;
          while mseAux > mse
              alfa = alfa * 0.9;
              Aaux = cell(L,1);
              for l=1:L
                  Aaux{l} = A{l} - alfa * deltaA{l};
              end
              Baux = B - alfa * deltaB;
              Y = MLPsaida( X(:,2:end), L, Aaux, Baux );
              err = (Y-Yd);
              mseAux = sum(sum( err.^2 ))/N;
          end
          alfa = alfa/0.9;
          A = Aaux;
          B = Baux;
          
          mseAux2 = mse;
          mse = mseAux;
          mseAux = mseAux2;
          %}
      end
      
      
      %Stores some values for plotting
      MSEtrain = [MSEtrain;mse];
      MSEval = [MSEval;mseVal];
      
      %{
      fig=figure(1);
      clf(fig);
      plot(MSEtrain, 'b');
      hold on
      plot(MSEval, 'r');
      title(sprintf('EQM X Epocas (h=%d, mE=%d)',h,mE)); 
      xlabel('Epocas');
      ylabel('Erro Quadratico Medio');            
      legend('train', 'validation');
      %}
      %{
      alfas = [alfas;alfa];
      fig = figure(2);
      clf(fig);
      plot(alfas);
      %}
      %fprintf('train = %.8f\tval = %.8f\n',mse,mseVal);
      %fprintf('%.8f\n',alfa);
    end

    %close(figure(1));
    %close(figure(2));
end