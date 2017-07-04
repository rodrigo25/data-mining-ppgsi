function [A, B, MSEtrain, MSEval, epochsExecuted] = MLPtreina(X, Yd, Xval, Ydval, h, mE, alfaType, gradientType, earlyStopping, alfa0)
% MLPtreina treina uma MLP com 1 camada oculta para um dataset de N linhas,
% ne colunas e nc classes.
% X         - Training dataset, matriz (N,ne)
% Yd        - Training labels (N,nc)
% Xval      - Validation dataset
% Ydval     - Validation labels
% h         - Nº neuronios em cada camada oculta
% mE        - max iterations
% alfa0     - Taxa de aprendizado inicial
% alfaType     -
%                   0: constante, sem decaimento
%                   1: decaimento pela metade step (a cada 5 epocas)
%                   2: bisseção
% gradientType -
%                   0: simples
%                   1: conjugado
% earlyStopping -
%                   0: nao para o treino se o MSE de validacao subir
%                   1: para o treino se o MSE de validacao subir
    if nargin < 9
        if alfaType ~= 2
            error('Alfa must be defined if when bisection method is not used');
        end
    end
    
    [N, ne] = size(X);
    [~, ns] = size(Yd);
    
    %Inicializa
    A = rand(h, ne+1)/10;
    B = rand(ns, h+1)/10; 
    [Y, ~] = MLPsaida( X, A, B );
	%Erro inicial
    err = Y-Yd;
    mse = sum(sum( err.^2 ))/N;

    MSEtrain = [];
    MSEval = [];
    alfas = [];
    minAlfa = .001;
    
    earlyStoppingEpochsNum = 20;
    
    epochsExecuted = 0;
    it = 0;
    while it < mE && mse > 1e-5
      it = it + 1;
      
      %Feedfoward
      [Y, Z] = MLPsaida( X, A, B );

      %Training error
      err = Y-Yd;
      mse = sum(sum( err.^2 ))/N;
    
      %Validation error
      Yval = MLPsaida(Xval, A, B);
      errVal = Yval-Ydval;
      mseVal = sum(sum( errVal.^2 ))/size(Xval,1);
     
      %Saves A, B to restore (early stopping case)
      if mod(it,earlyStoppingEpochsNum) == 0
         savedA = A;
         savedB = B;
      end

      % if early stopping is enabled
      % and if the last epochs show no improvement
      if earlyStopping && mod(it-1,earlyStoppingEpochsNum) == 0 && (it-5>0) && mseVal >= MSEval(it-5)
          % check last epochs improvements
          % assume no improvement
          lastEpochsNoImprovement = 1;
          for i=5:5:earlyStoppingEpochsNum
              if it-i <= 0 || it-(i+5) <= 0
                 % not enough epochs to check
                 break; 
              end
              if MSEval(it-i) < MSEval(it-(i+5))
                  % found some improvement
                  lastEpochsNoImprovement = 0;
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
      
      %Backpropagation
      gradB = err.*((1-Y).*Y);
      gradA = (gradB*B(:,2:end)).*(Z.*(1-Z));

      %The 2 scenarios where it's nice to reduce the size of gradients
      if alfaType == 2 || gradientType == 1
          gradB = gradB * 1/N;
          gradA = gradA * 1/N;          
      end
      
      if gradientType == 1
          % gradiente conjugado polak-ribiere
          if it == 1
              gradB_old = gradB;
              gradA_old = gradA;

              normA = zeros(h,1);
              for i=1:h
                 normA(i) = gradA(:,i)' * gradA(:,i);
              end
              normB = zeros(ns,1);
              for i=1:ns
                 normB(i) = gradB(:,i)' * gradB(:,i);
              end
              
              gB = sqrt(normB');
              gA = sqrt(normA');
              gradB = gradB ./ repmat(gB+0.000001,N,1);
              gradA = gradA ./ repmat(gA+0.000001,N,1);
          else
              newNormA = zeros(h,1);
              for i=1:h
                 newNormA(i) = gradA(:,i)' * gradA(:,i);
              end
              newNormB = zeros(ns,1);
              for i=1:ns
                 newNormB(i) = gradB(:,i)' * gradB(:,i);
              end
              
              if mod(it,ns) ~= 0
                  beta = ((gradB - gradB_old)'*gradB)./repmat(gB+0.000001,ns,1);
                  gradB = gradB + gradB * beta;
              end
                            
              if mod(it,h) ~= 0
                  beta = ((gradA - gradA_old)'*gradA)./repmat(gA+0.000001,h,1);
                  gradA = gradA + gradA * beta;
              end

              normB = newNormB;
              normA = newNormA;
                            
              gradB_old = gradB;
              gradA_old = gradA;
              
              gB = sqrt(normB');
              gA = sqrt(normA');
              gradB = gradB ./ repmat(gB+0.000001,N,1);
              gradA = gradA ./ repmat(gA+0.000001,N,1);
          end 
      end
      
      if alfaType == 2 % if bisection
         alfa = calcAlfa(X, Z, Yd, A, B, gradA, gradB, N); 
      else
         alfa = alfa0;
      end
      
      deltaA = alfa * gradA' * [ones(N,1) X];
      deltaB = alfa * gradB' * [ones(N,1) Z];

      A = A - deltaA;
      B = B - deltaB;

      
      %Learning rate update
      if alfaType == 0
          alfa = alfa0;
      elseif alfaType == 1 && mod(it,5) == 0 && alfa > minAlfa
          alfa = .5 * alfa;
      end
      
      %Stores some values for plotting
      MSEtrain = [MSEtrain;mse];
      MSEval = [MSEval;mseVal];
      alfas = [alfas alfa];  
      
      epochsExecuted = it;
      
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

function alfa_m = calcAlfa(X, Z, Yd, A, B, gradA, gradB, N)

    gB = gradB' * [ones(N,1) Z];
    gA = gradA' * [ones(N,1) X];
    g = [gA(:);gB(:)];
    d = -g;
    
    alfa_l = 0;
    alfa_u = rand;
    Aaux = A - alfa_u * gA;
    Baux = B - alfa_u * gB;
    g  = calcGrad(X, Yd, Aaux, Baux, N);
    hl = g'*d;

    while hl<0 
        alfa_u = alfa_u * 2;
        Aaux = A - alfa_u * gA;
        Baux = B - alfa_u * gB;
        g = calcGrad(X, Yd, Aaux, Baux, N);
        hl = g'*d;
    end
    
    epsilon = 1e-8;
    alfa_m = alfa_u;
    maxIt = ceil(log(alfa_u/epsilon));
    
    it = 0;
    while abs(hl)>epsilon && it < maxIt
    
        it = it + 1;
        alfa_m = (alfa_l+alfa_u)/2;
        Aaux = A - alfa_m * gA;
        Baux = B - alfa_m * gB;
        g = calcGrad(X, Yd, Aaux, Baux, N);
        hl = g'*d;

        if hl<0
            alfa_l = alfa_m;
        else 
            alfa_u = alfa_m;
        end
    end
end

function [g, gradA, gradB] = calcGrad(X, Yd, A, B, N)
    
    [Y, Z] = MLPsaida( X, A, B );

    err = Y-Yd;

    gradB = 1/N * err.*((1-Y).*Y);
    gradA = 1/N * (gradB*B(:,2:end)).*(Z.*(1-Z));
    
    gB = gradB' * [ones(N,1) Z];
    gA = gradA' * [ones(N,1) X];
    g = [gA(:);gB(:)];
end