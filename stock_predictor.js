const LogisticRegression = require('ml-logistic-regression');
const csv = require('csv-parser');
const fs = require('fs');
const { Matrix } = require('ml-matrix');

async function trainModel() {
    const data = [];
    const features = [];
    const labels = [];

    // Ler o arquivo CSV
    fs.createReadStream('GoogleData.csv')  // Substitua pelo seu arquivo CSV real
        .pipe(csv())
        .on('data', (row) => {
            // Supondo que suas colunas no CSV sejam 'feature1', 'feature2' e 'label'
            features.push([parseFloat(row.Adj_Close), parseFloat(row.Close)]); // Ajuste de acordo com seu CSV
            labels.push(parseInt(row.Volume));  // Ajuste para os rótulos
        })
        .on('end', () => {
            const trainSize = Math.floor(features.length * 0.8);  // 80% treino, 20% teste
            const X_train = new Matrix(features.slice(0, trainSize));
            const X_test = new Matrix(features.slice(trainSize));
            const Y_train = Matrix.columnVector(labels.slice(0, trainSize));
            const Y_test = Matrix.columnVector(labels.slice(trainSize));

            // Instanciar o modelo de regressão logística
            const logistic = new LogisticRegression({
                numSteps: 1000,      // Número de iterações
                learningRate: 5e-3   // Taxa de aprendizado
            });

            // Treinar o modelo com os dados
            logistic.train(X_train, Y_train);

            console.log('Modelo treinado com sucesso');

            // Passo 1: Fazer previsões com os dados de teste
            const predictions = logistic.predict(X_test);

            // Exibir as previsões e os rótulos reais
            console.log('Previsões:');
            console.log(predictions);

            console.log('Rótulos verdadeiros:');
            console.log(Y_test.to1DArray());  // Converter para uma array para melhor exibição
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
