function [  ] = create_log( res, dirName, redeConfig, fN)

if exist('Resultados/', 'dir') == 0 % se o dir nao existe, ele eh criado
  mkdir('Resultados/');
end

file_name = [dirName 'log' num2str(fN) '.txt'];
fileID = fopen(file_name, 'w');

fprintf(fileID, '\n\t\t\t Log de Teste da MLP\n\n');

fprintf(fileID, '\n\n ---------------------------------------------------\n');
fprintf(fileID, '                 Configurações da Rede              \n');
fprintf(fileID, ' ---------------------------------------------------\n');
fprintf(fileID, ' Nome: \t\t\t\t\t\t\t%s\n', redeConfig.Nome);
fprintf(fileID, ' Função de Otimização: \t\t\t%s\n', redeConfig.Otimizacao);
fprintf(fileID, ' Camadas Escondidas: \t\t\t%d\n', redeConfig.Camadas);
fprintf(fileID, ' Neurônios na Camada Escondida: %d\n', redeConfig.Neuronios);
fprintf(fileID, ' Alfa: \t\t\t\t\t\t\t%s\n', redeConfig.Alfa);
fprintf(fileID, ' Máximo de Iterações: \t\t\t%d\n', redeConfig.MaxIt);
fprintf(fileID, ' Limiar de Decisão: \t\t\t%f\n', redeConfig.Limiar);


fprintf(fileID, '\n\n ---------------------------------------------------\n');
fprintf(fileID, '                  Matriz de Confusão                \n');
fprintf(fileID, ' ---------------------------------------------------\n');
fprintf(fileID, '\n\t            Predicted         \n');
fprintf(fileID, '\t      ----------------------  \n');
fprintf(fileID, '\t     |     P    |      N    |     \n');
fprintf(fileID, '\t  --------------------------|\n');
fprintf(fileID, '\t | P |\t%4.0d\t|\t%4.0d\t|    \n',res(2).TP,res(2).FN);
fprintf(fileID, '\t  -------------------------- \n');
fprintf(fileID, '\t | N |\t%4.0d\t|\t%4.0d\t|    \n',res(2).FP,res(2).TN);
fprintf(fileID, '\t  -------------------------- \n');

fprintf(fileID, '\n%s \t\t- %d\n', res(1).TP, res(2).TP);
fprintf(fileID, '%s \t\t- %d\n', res(1).FN, res(2).FN);
fprintf(fileID, '%s \t\t- %d\n', res(1).FP, res(2).FP);
fprintf(fileID, '%s \t\t- %d\n', res(1).TN, res(2).TN);

fprintf(fileID, '\n\n ---------------------------------------------------\n');
fprintf(fileID, '                    Estatísticas                    \n');
fprintf(fileID, ' ---------------------------------------------------\n');
fprintf(fileID, '%s \t\t\t- %f\n', res(1).TPR, res(2).TPR);
fprintf(fileID, '%s \t\t- %f\n', res(1).TNR, res(2).TNR);
fprintf(fileID, '%s \t- %f\n', res(1).PPV, res(2).PPV);
fprintf(fileID, '%s \t\t\t\t- %f\n', res(1).NPV, res(2).NPV);
fprintf(fileID, '%s \t\t- %f\n', res(1).FNR, res(2).FNR);
fprintf(fileID, '%s \t\t\t\t\t- %f\n', res(1).FDR, res(2).FDR);
fprintf(fileID, '%s \t\t\t\t\t- %f\n', res(1).FOR, res(2).FOR);
fprintf(fileID, '%s \t\t\t\t\t\t\t\t- %f\n', res(1).ACC, res(2).ACC);
fprintf(fileID, '%s \t\t\t\t\t\t\t\t- %f\n', res(1).Fscore, res(2).Fscore);
fprintf(fileID, '%s \t\t- %f\n', res(1).MCC, res(2).MCC);
fprintf(fileID, ' ---------------------------------------------------\n');



fclose(fileID);

end

