function Yh = adaboostM2(Xtr, Ytr, Xtest, Ytest, classes, T, h, nepocasMax, resampleMode)
    
    if size(Ytr,2) ~= length(classes) || size(Ytest,2) ~= length(classes)
       error('Training or test labels dimensions dont correspond to num of classe'); 
    end

    [A, B, beta] = train(T, Xtr, Ytr, Xtest, Ytest, classes, h, nepocasMax, resampleMode);
    
    Yh = 0;
    for t=1:T
       Yh = Yh + log(1/beta(t))*result(Xtest, A{t}, B{t});
    end
    sum(Ytr)
end

function [A, B, beta, D] = train(T, Xtr0, Ytr0, Xt, Yt, classes, h, epochs, resampleMode)
    
% Notacao
% N - num de instancias
% ne - num de entradas
% nc - num de classes
% ns - num de saidas
% h - num de neuronios na camanda oculta

    % Local variables initializations
    nc = length(classes);
    
    Xtr = Xtr0;
    Ytr = Ytr0;
    
    [N,ne] = size(Xtr);
    ns = size(Ytr,2);
    
    % Output MLP components weigths
    % A - input layer to hidden layer
    % B - hidden layer to output layer
    A = cell(T);
    B = cell(T);
    
    D = repmat(1/(N * nc-1),N,nc-1);
    beta = zeros(T,1);
    
    t = 0;
    while t<T
        t = t+1;       
        if t > 1 && resampleMode == 1
            [Xtr, Ytr, D] = resample(Xtr0, Ytr0, D);
        end
        nPerClass = sum(Ytr)
        V = ones(N,nc);
        V(Ytr ~= 1) = D./repmat((max(D,[],2)),1,nc-1);

        % Trains the MLP 
        [A_t,B_t] = MLP(Xtr,Ytr,h,epochs,V);
        % Yh - MLPs hypothesis
        Yh = result( Xtr, A_t, B_t );
        
        % Discard classifier if accuracy is too low
        [~,Yh_] = max(Yh,[],2);
        [~,Ytr_] = max(Ytr,[],2);
        acc = multiclassConfusionMatrix(Ytr_, Yh_, classes, sprintf('T=%d (training set)', t));
        [~,majorClass] = max(nPerClass);
        nMajorClass = nPerClass(majorClass);
        if acc <= (nMajorClass / sum(nPerClass))
           %t = t - 1;
           %fprintf('Low accuracy, discarding classifier');
           %continue;
        end
        
        %Yh_term - hypothesis confidence on the right labels
        Yh_term = repmat(max(Ytr .* Yh,[],2),1,nc-1);
        
        %Yh_term_miss - hypothesis confidence on the miss-labels
        Yh_term_miss = zeros(N,nc-1);
        for i=1:N
            [~,labelInd] = max(Ytr(i,:),[],2);
            Yh_term_miss(i,:) = [Yh(i,1:labelInd-1) Yh(i,labelInd+1:end)];
        end
        
        epsilon_t = 1/2 * sum(sum( D .* (1 - Yh_term + Yh_term_miss) ));
        beta_t = epsilon_t / (1 - epsilon_t);
        
        fprintf('Adaboost round %d epsilon=%f\n', t, epsilon_t);
        
        D = D .* (beta_t .^ ( 1/2 *(1 + Yh_term - Yh_term_miss) ));
        %D = normalization(D, 'minmax');
        
        A{t} = A_t;
        B{t} = B_t;
        beta(t) = beta_t;
    end
end

function [Xtr, Ytr, D] = resample(Xtr, Ytr, D)
%Resampling based on prob distribution generated from D
    [N,~] = size(Xtr);
    P = sum(D,2);
    P = P ./ sum(P);
    resample = 0;
    while resample < N
        for i=1:N
           if rand < P(i)
               resample = resample + 1;
               Xtr(resample,:) = Xtr(i,:);
               Ytr(resample,:) = Ytr(i,:);
               D(resample,:) = D(i,:);
               if resample == N
                  break; 
               end
           end
        end
    end
end

function Y = result(X, A, B)
    N = size(X,1);
    X = [ones(N,1),X];
    Zin = X*A';
    Z = 1./(1+exp(-Zin));
    Yin = [ones(N,1),Z]*B';
    Y = 1./(1+exp(-Yin));
end

function [A, B] = MLP(X, Yd, h, mE, V)

    [N, ne] = size(X);
    ns = size(Yd,2);
    X = [ones(N,1),X];

    A = rand(h, ne+1); 
    B = rand(ns, h+1); 

    alfa0 = .01;
    maxEqm = 1e-5;

    EQM = [];
    alfas = alfa0;
    alfa = alfa0;

    fig = figure;
    for it=1:mE

      %Feedfoward
      Zin = X*A';
      Z =  1./(1+exp(-Zin));
      Yin = [ones(N,1),Z]*B';
      Y = 1./(1+exp(-Yin));

      %calcula o erro
      err = V.*(Y-Yd);
      eqm = sum(sum( err.^2 ))/N;
      if eqm<maxEqm
        break
      end;

      %Backpropagation
      gradB = err.*(1-Y).*Y;
      gradA = (gradB*B(:,2:end)).*(Z.*(1-Z));

      deltaA = alfa * gradA' * X;
      deltaB = alfa * gradB' * [ones(N,1),Z];

      %Weights update
      A = A - deltaA;
      B = B - deltaB;

      %Learning rate update
      %alfa = alfa0/(1 + it*0.05);
      alfa = alfa0;

      %Stores some values for plotting
      alfas = [alfas;alfa];
      EQM = [EQM;eqm];

       clf(fig);
       %plot(Ytr,'r--')
       %hold on
       %plot(Y,'b')
       %grid
       plot(EQM);
       title('EQM X Epocas');
       xlabel('Epocas');
       ylabel('Erro Quadratico Medio');
       text(it, eqm, sprintf('it=%d',it));
    end

    close(fig);
end