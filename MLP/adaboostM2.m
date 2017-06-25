function Yh = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest,classes, T, h, nepocasMax, discardMode, resampleMode, nnArchMode)
% adaboostM2 pseudo-loss adaboost m2 (multi-class)
% Xtr           - training set (N,ne)
% Ytr           - training set labels, binary matrix (N,nc) nc = num classes
% Xtest, Ytest  - test set, test set labels
% classes       - dataset labels, length(classes) = nc
% T             - Num of adaboost rounds
% h             - hidden layer neurons count
% nepocasMax    - max num of epochs
% discardMode  - 
%                   0: discards the t-th classifier if pseudo-loss >= .5
%                   1: discards the t-th classifier if guessing by major
%                   class is better
% resampleMode - 
%                   0: no resample
%                   1: resample before training t-th classifier
% nnArchMode    - 
%                   0: constant MLP architecture
%                   1: h increased by 10 at each round t
% 

    
    % Validate input 
    if size(Ytr,2) ~= length(classes) || size(Ytest,2) ~= length(classes)
       error('Training or test labels dimensions dont correspond to num of classe'); 
    end
    
    if resampleMode~=0 && resampleMode~=1
       error('Resample mode should be 0 or 1'); 
    end
    
    if discardMode~=0 && discardMode~=1
       error('Discard mode should be 0 or 1'); 
    end
    
    if nnArchMode~=0 && nnArchMode~=1
       error('NN arch mode should be 0 or 1'); 
    end
    % end validating input

    [A, B, beta, P] = train(T, Xtr, Ytr, Xval, Yval, classes, h, nepocasMax, discardMode, resampleMode, nnArchMode);
    
    Yh = 0;
    for t=1:T
       Yh = Yh + log(1/beta(t))*MLPsaida([Xtest repmat(1/size(Xtest,1),size(Xtest,1),1)], A{t}, B{t});
       %Yh = Yh + log(1/beta(t))*MLPsaida(Xtest, A{t}, B{t});
    end
    sum(Ytr)
end

function [A, B, beta, P] = train(T, Xtr0, Ytr0, Xval, Yval, classes, h, epochs, discardMode, resampleMode, nnArchMode)
    
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
    
    P = cell(T);
    
    % D - Distribution over miss-labels 
    % miss-labels = {(i,y)| i in {1;...;N} & y ~= yi }
    D = repmat(1/(N * nc-1),N,nc-1);
    beta = zeros(T,1);
    
    lastResampledRound = 1; % resample only after t > 1
    lastNNArchChangedRound = 1; % change arch only after t > 1
    
    t = 0; %indicates current Adaboost round
    while t<T
        t = t+1;       
        fprintf('Adaboost round %d\n', t);
        if resampleMode == 1 && lastResampledRound ~= t
            [Xtr, Ytr, D] = resample(Xtr0, Ytr0, D);
            lastResampledRound = t;
        end
        P_t = sum(D,2);
        P_t = P_t ./ sum(P_t);
        P{t} = P_t;
        
        nPerClass = sum(Ytr)
        
        % Changes MLP arch if needed
        if nnArchMode==1 && lastNNArchChangedRound~=t
           h = h + 10;
           %epochs = epochs + 250;
           lastNNArchChangedRound=t;
        end
        
        V = ones(N,nc);
        V(Ytr ~= 1) = D./repmat((max(D,[],2)),1,nc-1);
        
        % Trains the MLP 
        [A_t,B_t] = MLPtreina([Xtr P_t],Ytr,[Xval repmat(1/size(Xval,1), size(Xval,1),1)],Yval,h,epochs,.01,0,0,V);
        %[A_t,B_t] = MLPtreina(Xtr,Ytr,Xval,Yval,h,epochs,.01,0,1,V);
        
        % Yh - MLPs hypothesis
        Yh = MLPsaida([Xtr P_t], A_t, B_t);
        %Yh = MLPsaida(Xtr, A_t, B_t);
        
        % Discard classifier if accuracy is too low
        if discardMode == 1
            [~,Yh_] = max(Yh,[],2);
            [~,Ytr_] = max(Ytr,[],2);
            acc = multiclassConfusionMatrix( Ytr_, Yh_, classes, 1, sprintf('T=%d (training set)', t) );
            [~,majorClass] = max(nPerClass);
            nMajorClass = nPerClass(majorClass);
            majorClassAcc = (nMajorClass / sum(nPerClass));
            if acc <= majorClassAcc
               t = t - 1;
               fprintf('Low accuracy, discarding classifier. Random guessing would have acc=%f\n', majorClassAcc);
               continue;
            end 
        end
        
        % Calculates pseudo-loss
        
        %Yh_term - hypothesis confidence on the right labels
        Yh_term = repmat(max(Ytr .* Yh,[],2),1,nc-1);
        
        %Yh_term_miss - hypothesis confidence on the miss-labels
        Yh_term_miss = zeros(N,nc-1);
        for i=1:N
            [~,rightLabel] = max(Ytr(i,:),[],2);
            Yh_term_miss(i,:) = [Yh(i,1:rightLabel-1) Yh(i,rightLabel+1:end)];
        end
        
        % pseudo-loss(t) = epsilon_t
        epsilon_t = 1/2 * sum(sum( D .* (1 - Yh_term + Yh_term_miss) ));
        beta_t = epsilon_t / (1 - epsilon_t);
        
        % Discards if pseudo-loss is too high
        if discardMode==0 && epsilon_t >= .5
            fprintf('Pseudo-loss too high %f discarding classifier %d\n', epsilon_t, t);
            t = t - 1;
            continue;
        end
        
        fprintf('Adaboost round %d epsilon=%f\n', t, epsilon_t);
        
        D = D .* (beta_t .^ ( 1/2 *(1 + Yh_term - Yh_term_miss)));
        
        %{
        [~,Ytrc] = max(Ytr,[],2);
        [~,Yhc] = max(Yh,[],2);
        D(Ytrc ~= Yhc,:) = 10 * D(Ytrc ~= Yhc,:);
        D(Ytrc == Yhc,:) = D(Ytrc == Yhc,:) ./ 10;
        %}
        
        %Normalizes so sum(sum(D)) = 1
        D = D ./ sum(sum(D));
        
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