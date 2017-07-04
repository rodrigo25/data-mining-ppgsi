function [A, B, MSEtrain, MSEval, epochsExecuted, Yc] = MLPtreinaRegrClassif(X, Yd, Xval, Ydval, Xtest, ...
        h, mE, alfaType, gradientType, earlyStopping, alfa0)
    
    Yd0 = Yd;
    Ydval0 = Ydval;
    [~,Yd] = max(Yd,[],2);
    [~,Ydval] = max(Ydval,[],2);
    
    minY = min(Yd);
    maxY = max(Yd);
    
    % Tranforma de problema multiclasse para regressao
    % colocando as classes entre 0 e 1, dado que a sigmoide
    % na camada de saida somente trabalha nessa faixa
    
    % simplesmente divide por 10 (6 classes nos datasets, no max)
    %Yd = Yd / 10;
    %Ydval = Ydval / 10;
    
    % usa minmax
    Yd = normalization(Yd,'minmax');
    Ydval = normalization(Ydval,'minmax');
    
    [A, B, MSEtrain, MSEval, epochsExecuted] = MLPtreina(X, Yd, Xval, Ydval, ...
        h, mE, alfaType, gradientType, earlyStopping, alfa0);
    
    Yc = MLPsaida( Xtest, A, B );
    
    %Yc = Yc * 10;
    
    % Desfaz o minmax para converter para a faixa original de valores
    Yc = (Yc + minY)*(maxY - minY);
    
    Yc = round(Yc);
    
    % Converte para problema multiclasse
    Yc = multiclassY(Yc);
end