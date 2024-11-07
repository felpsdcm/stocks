const LogisticRegression = require('ml-logistic-regression');  // Importação corrigida
const csv = require('csv-parser');
const fs = require('fs');
const { Matrix } = require('ml-matrix');

async function trainModel() {
    const data = [];

    // Ler o arquivo CSV
    fs.createReadStream('GoogleData.csv')  // Substitua 'your_data.csv' pelo nome correto do arquivo CSV
        .pipe(csv())
        .on('data', (row) => {
            data.push(row); // Adicione lógica para processar cada linha
        })
        .on('end', () => {
            // Exemplo de como preparar os dados para features e labels
            const features = [
                [1, 2],  // Exemplo de dados (substitua pelos dados reais)
                [3, 4],
                [5, 6]
            ]; 

            const labels = [0, 1, 0]; // Exemplo de rótulos (substitua pelos rótulos reais)
            
            // Criar as matrizes para o modelo
            const X = new Matrix(features);  // Matriz de features
            const Y = Matrix.columnVector(labels);  // Matriz de labels

            // Instanciar o modelo de regressão logística
            const logistic = new LogisticRegression({
                numSteps: 1000,      // Número de iterações
                learningRate: 5e-3   // Taxa de aprendizado
            });

            // Treinar o modelo com os dados
            logistic.train(X, Y);

            console.log('Modelo treinado com sucesso');
        });

        
}

async function constructModel() {
    try {
        await trainModel();
    } catch (error) {
        console.error('Erro ao treinar o modelo:', error);
    }
}



constructModel();




