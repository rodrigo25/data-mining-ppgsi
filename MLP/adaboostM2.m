function [Yh, A, B, beta, MSEtrain, MSEval] = adaboostM2(Xtr, Ytr, Xval, Yval, Xtest, Ytest,classes, T, h, nepocasMax, nnArchMode)
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

    [A, B, beta, MSEtrain, MSEval] = train(T, Xtr, Ytr, Xval, Yval, classes, h, nepocasMax, nnArchMode);
    
    Yh = 0;
    for t=1:T
       Yh = Yh + log(1/beta(t))*MLPsaida(Xtest, A{t}, B{t});
    end
end

function [A, B, beta, MSEtrain, MSEval] = train(T, Xtr0, Ytr0, Xval, Yval, classes, h, epochs, nnArchMode)
    
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

    MSEtrain = zeros(T,1);
    MSEval = zeros(T,1);
    % D - Distribution over miss-labels 
    % miss-labels = {(i,y)| i in {1;...;N} & y ~= yi }
    D = repmat(1/(N * nc-1),N,nc-1);
    beta = zeros(T,1);
    
    lastResampledRound = 1; % resample only after t > 1
    lastNNArchChangedRound = 1; % change arch only after t > 1
    
    t = 0; %indicates current Adaboost round
    fprintf('Adaboost rounds:\n')
    while t<T
        t = t+1;
        
        fprintf('%d ', t);
        if mod(t,20) == 0
            fprintf('\n');
        end
        %fprintf('Adaboost M2 round %d\n', t);
        if lastResampledRound ~= t
            [Xtr, Ytr] = resample(Xtr0, Ytr0, D);
            lastResampledRound = t;
        end
        
        %disp('# per class');
        %disp(sum(Ytr));
        
        % Changes MLP arch if needed
        if nnArchMode==1 && lastNNArchChangedRound~=t
           % TODO mudar neuronios
           % h = round( rand * (13 - 7) ) + 7;
           lastNNArchChangedRound=t;
        end
                        
        %V = ones(N,nc);
        %V(Ytr ~= 1) = D./repmat((max(D,[],2)),1,nc-1);
        
        % Trains the MLP 
        [A_t,B_t, netMSEtrain, netMSEval, epochsExecuted] = MLPtreina(Xtr,Ytr,Xval,Yval,h,epochs,2,1,0);
        %[A_t,B_t] = MLP_clodoaldo(Xtr,Ytr,h,epochs);
        %[A_t,B_t] = MLP_alfaAdaptativo(Xtr,Ytr,Xval,Yval,h,epochs,0);
       
        %Yh - MLPs hypothesis
        Yh = MLPsaida(Xtr0, A_t, B_t);
        
        % shows accuracy
        [~,Yhc]= max(Yh,[],2);
        [~,Ytrc] = max(Ytr0,[],2);
        
        %acc = multiclassConfusionMatrix( Ytrc, Yhc, classes );
        %fprintf('Adaboost M2 component %d executed %d epochs with validation error %f and training acc %f\n', t, epochsExecuted, netMSEval(epochsExecuted), acc);
        
        % Calculates pseudo-loss
        
        %Yh_term - hypothesis confidence on the right labels
        Yh_term = repmat(max(Ytr0 .* Yh,[],2),1,nc-1);
        
        %Yh_term_miss - hypothesis confidence on the miss-labels
        Yh_term_miss = zeros(N,nc-1);
        for i=1:N
            [~,rightLabel] = max(Ytr0(i,:),[],2);
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
        
        %fprintf('Adaboost M2 round %d epsilon=%f\n', t, epsilon_t);
        
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
        MSEval(t) = netMSEval(epochsExecuted);
        MSEtrain(t) = netMSEtrain(epochsExecuted);
    end
end

function [Xtr, Ytr] = resample(Xtr, Ytr, D)
%Resampling based on prob distribution generated from D
    [N,~] = size(Xtr);
    P = sum(D,2);
    P = P ./ sum(P);
    resample = randsample(N,N,true,P);
    Xtr(:,:) = Xtr(resample,:);
    Ytr(:,:) = Ytr(resample,:);
end

function [XtrResampled, YtrResampled] = resampleKeepClassDistr(Xtr, Ytr, D)
%Resampling based on prob distribution generated from D keeping classes
%distributions
    [N,ne] = size(Xtr);
    [~,ns] = size(Ytr);
    nPerClass = sum(Ytr);
    
    XtrResampled = zeros(N,ne);
    YtrResampled = zeros(N,ns);
    
    index = 1;
    for i=1:size(Ytr,2)
       Ytr_i = Ytr( Ytr(:,i) == 1, :);
       Xtr_i = Xtr( Ytr(:,i) == 1, : );
       D_i = D( Ytr(:,i) == 1, : );
       P = sum(D_i,2);
       P = P ./ sum(P);
       N_i = nPerClass(i);
       resample = randsample(N_i,N_i,true,P);
       
       XtrResampled(index:(index + N_i - 1),:) = Xtr_i(resample,:);
       YtrResampled(index:(index + N_i - 1),:) = Ytr_i(resample,:);
       index = index + N_i;
    end
end