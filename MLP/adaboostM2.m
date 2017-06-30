function [Yh, XYtrained] = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest,classes, T, h, nepocasMax, nnArchMode)
% adaboostM2 pseudo-loss adaboost m2 (multi-class)
% Xtr           - training set (N,ne)
% Ytr           - training set labels, binary matrix (N,nc) nc = num classes
% Xtest, Ytest  - test set, test set labels
% classes       - dataset labels, length(classes) = nc
% T             - Num of adaboost rounds
% h             - hidden layer neurons count
% nepocasMax    - max num of epochs
% nnArchMode    - 
%                   0: constant MLP architecture
%                   1: MLP hidden layer count increased by 1 at each round t
     
    % Validate input 
    if T < 2
       error('Adaboost rounds must be >= 2'); 
    end
    if size(Ytr,2) ~= length(classes) || size(Ytest,2) ~= length(classes)
       error('Training or test labels dimensions dont correspond to num of classe'); 
    end
    
    if nnArchMode~=0 && nnArchMode~=1
       error('NN arch mode should be 0 or 1'); 
    end
    % end validate input

    [A, B, beta, L, XYtrained] = train(T, Xtr, Ytr, Xval, Yval, classes, h, nepocasMax, nnArchMode);
    
    Yh = 0;
    for t=1:T
       Yh = Yh + log(1/beta(t))*MLPsaida(Xtest, L(t), A{t}, B{t});
    end
end

function [A, B, beta, L, XYtrained ] = train(T, Xtr0, Ytr0, Xval, Yval, classes, h, epochs, nnArchMode)
    
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
    A = cell(T,1);
    B = cell(T,1);
    
    P = cell(T,1);
    
    XYtrained = cell(T,2);
    
    % D - Distribution over miss-labels 
    % miss-labels = {(i,y)| i in {1;...;N} & y ~= yi }
    D = repmat(1/(N * nc-1),N,nc-1);
    beta = zeros(T,1);
    
    lastResampledRound = 1; % resample only after t > 1
    lastNNArchChangedRound = 1; % change arch only after t > 1
    
    L = ones(T,1); %MLP hidden layers, inicializadas com 1
    t = 0; %indicates current Adaboost round
    while t<T
        t = t+1;       
        fprintf('Adaboost round %d\n', t);
        if lastResampledRound ~= t
            [Xtr, Ytr, D] = resample(Xtr0, Ytr0, D);
            lastResampledRound = t;
        end
        P_t = sum(D,2);
        P_t = P_t ./ sum(P_t);
        P{t} = P_t;
        
        XYtrained{t,1} = Xtr;
        XYtrained{t,2} = Ytr;
        
        nPerClass = sum(Ytr)
        
        % Changes MLP arch if needed
        if nnArchMode==1 && lastNNArchChangedRound~=t
           %L(t) = L(t) + mod(t-1,2);
           %epochs = epochs - 100;
           lastNNArchChangedRound=t;
        end
        
        V = ones(N,nc);
        %V(Ytr ~= 1) = D./repmat((max(D,[],2)),1,nc-1);
        
        % Trains the MLP 
        [A_t,B_t] = MLPtreina(Xtr,Ytr,Xval,Yval,L(t),h,epochs,.01,0,0,V);
        %[A_t,B_t] = treina_rede(Xtr,Ytr,h,epochs);
        %[A_t,B_t,~] = treinamento(Xtr,Ytr,h,epochs);
        
        %Yh - MLPs hypothesis
        Yh = MLPsaida(Xtr, L(t), A_t, B_t);
        
        % shows accuracy
        [~,Yhc]= max(Yh,[],2);
        [~,Ytrc] = max(Ytr,[],2);
        multiclassConfusionMatrix( Ytrc, Yhc, classes );
        
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
        if epsilon_t >= .5
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