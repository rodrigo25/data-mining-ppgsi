function [ output_args ] = create_log( confMatrix, estatisticas )

if exist('Resultados/', 'dir') == 0 % se o dir nao existe, ele eh criado
  mkdir('Resultados/');
end
dt = datetime;
dt.Format = 'yyyy-MM-dd''T''HH-mm-ss';

file_name = ['Resultados/' char(dt) '.txt'];
%file_name = strcat('holdout/resumo_', algoritmo, '_', tipo, '_', num2str(qtdPessoas), );

fileID = fopen(file_name, 'w');
%fprintf(fileID, '\nTempo de treinamento: %d\n\n', tempos(1));

fprintf(fileID, ' ---------------------------------------\n');
for i=1:15
  fprintf(fileID, '%s \t- %f\n', estatisticas{i,2}, estatisticas{i,3});
end
fprintf(fileID, ' ---------------------------------------\n');

fclose(fileID);

end

