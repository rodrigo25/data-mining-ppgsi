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
        
        fprintf('%d\n', t);
        if mod(t,10) == 0
            %fprintf('\n');
        end
        %fprintf('Adaboost M2 round %d\n', t);
        if lastResampledRound ~= t
            
            %if lastResampledRound == 1
            %    [Xtr, Ytr, D, resample] = resample3(Xtr0, Ytr0, D);
            %else
            %    [Xtr, Ytr, D, resample] = resample3(Xtr0, Ytr0, D,resample);
            %end
            
            %[Xtr, Ytr] = resample2(Xtr0, Ytr0, D,t);
            [Xtr, Ytr] = resample(Xtr0, Ytr0, D);
            lastResampledRound = t;
        end
        
        disp('# per class');
        disp(sum(Ytr));
        
        % Changes MLP arch if needed
        if nnArchMode==1 && lastNNArchChangedRound~=t
           h = round(rand * (25 - 15) + 15);
           lastNNArchChangedRound=t;
        end
                        
        % Trains the MLP 
        [A_t,B_t, netMSEtrain, netMSEval, epochsExecuted] = MLPtreina(Xtr,Ytr,Xval,Yval,h,epochs,2,1,0);
        %[A_t,B_t, netMSEtrain, netMSEval, epochsExecuted] = MLPtreina(Xtr,Ytr,Xval,Yval,h,epochs,0,0,0,.01);
        %[A_t, B_t, netMSEtrain, netMSEval, ~] = MLP_alfaAdaptativo(Xtr,Ytr,Xval,Yval,h,epochs,0);
        %epochsExecuted = length(netMSEtrain);
        
        %Yh - MLPs hypothesis
        Yh = MLPsaida(Xtr0, A_t, B_t);
        
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
    %{
    [rows,~] = find(D > 0.5);
    P(rows) = P(rows) * 100;
    %}
    P = P ./ sum(P);
    
    resample = randsample(N,N,true,P);
    Xtr(:,:) = Xtr(resample,:);
    Ytr(:,:) = Ytr(resample,:);
    
    %{
    figure(1);
    clf;
    bar(1:size(Ytr,2),sum(Ytr));
    title('Class distribution');

    
    [~,Yhc] = max(Yh,[],2);
    [~,Ytrc] = max(Ytr,[],2);
    missed = find((Yhc ~= Ytrc));
    resampledMissedCount = zeros(1,numel(unique(Ytrc))); 
    for i=1:numel(missed)
       for j=1:N
            if missed(i) == resample(j)
                resampledMissedCount(Yhc(missed(i))) = resampledMissedCount(Yhc(missed(i))) + 1;
            end
        end 
    end
    disp('Resample selected:')
    disp(resampledMissedCount);
    %}
end

function [Xtr, Ytr, D] = resample2(Xtr, Ytr, D, Xtr0, Ytr0, t)
%Resampling based on prob distribution generated from D
%and based on class distribution weights
    
    [N,nc] = size(Ytr0);
    if mod(t,10) == 0
        %reset
        Xtr = Xtr0;
        Ytr = Ytr0;
        D = repmat(1/(N * nc-1),N,nc-1);
        return;
    end
    nPerClass = sum(Ytr);
    [~,Ytrc] = max(Ytr,[],2);
    
    W = N * (nPerClass(Ytrc)'.^(-1));

    P = sum(D,2);
    P = P .* W;
    P = P ./ sum(P);
    resample = randsample(N,N,true,P);
    Xtr(:,:) = Xtr(resample,:);
    Ytr(:,:) = Ytr(resample,:);
    
    figure(1);
    clf;
    bar(1:size(Ytr,2),sum(Ytr));
    title('Class distribution');
end

function [Xtr, Ytr, D, resample] = resample3(Xtr, Ytr, D0, resample0)
%Resampling based on prob distribution generated from D, reorders D as
%needed
    [N,nc] = size(Ytr);

    if nargin < 4
        D=D0;
    else 
        D = repmat(1/(N * nc-1),N,nc-1);
        for i=1:N
            ind = find(resample0 == i, 1, 'first');
            if ~isempty(ind)
               D(i,:) = D0(ind,:);
            end
        end
    end
    P = sum(D,2);
    P = P ./ sum(P);
    resample = randsample(N,N,true,P);
    Xtr(:,:) = Xtr(resample,:);
    Ytr(:,:) = Ytr(resample,:);
    
    figure(1);
    clf;
    bar(1:size(Ytr,2),sum(Ytr));
    title('Class distribution');
end

function [XtrResampled, YtrResampled] = resamplePermClassDistr(Xtr, Ytr, D)
%Resampling based on prob distribution generated from D keeping classes
%distributions
    [N,ne] = size(Xtr);
    [~,ns] = size(Ytr);
    nPerClass = sum(Ytr);
    
    %confWrongClasses = sum(D);
    nPerClass = nPerClass(randperm(numel(nPerClass)));
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
       resample = randsample(size(Xtr_i,1),N_i,true,P);
       
       XtrResampled(index:(index + N_i - 1),:) = Xtr_i(resample,:);
       YtrResampled(index:(index + N_i - 1),:) = Ytr_i(resample,:);
       index = index + N_i;
    end
    
    %figure(1);
    %clf;
    %bar(1:size(YtrResampled,2),sum(YtrResampled));
    %title('Class distribution');
end
    