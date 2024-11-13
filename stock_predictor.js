const LogisticRegression = require('ml-logistic-regression');
const csv = require('csv-parser');
const fs = require('fs');
const { Matrix } = require('ml-matrix');

async function trainModel() {
    const features = [];
    const labels = [];

    // Ler o arquivo CSV de forma assíncrona
    await new Promise((resolve, reject) => {
        fs.createReadStream('microsoft.csv')  // Substitua pelo seu arquivo CSV real
            .pipe(csv())
            .on('data', (row) => {
                // Verifique os dados lidos para garantir que são válidos
                console.log('Linha lida:', row);  // Adiciona este log para depuração

                // Certifique-se de que as colunas estão presentes e com valores válidos
                const adjClose = parseFloat(row.Adj_Close);
                const close = parseFloat(row.Close);
                const volume = parseInt(row.Volume);

                // Verifique se os valores são válidos
                if (!isNaN(adjClose) && !isNaN(close) && !isNaN(volume)) {
                    features.push([adjClose, close]);
                    labels.push(volume);
                } else {
                    console.warn(`Dados inválidos na linha: ${JSON.stringify(row)}`);
                }
            })
            .on('end', () => {
                resolve();
            })
            .on('error', reject);
    });

    console.log('Dados de treinamento lidos com sucesso.');

    // Normalizar os dados
    normalizeData(features);

    // Verifique os dados antes de treinar
    console.log('Features normalizadas:', features);
    console.log('Labels:', labels);

    // Dividir os dados em conjunto de treino e teste
    const trainSize = Math.floor(features.length * 0.8);  // 80% treino, 20% teste
    const X_train = new Matrix(features.slice(0, trainSize));
    const X_test = new Matrix(features.slice(trainSize));
    const Y_train = Matrix.columnVector(labels.slice(0, trainSize));
    const Y_test = Matrix.columnVector(labels.slice(trainSize));

    // Instanciar e treinar o modelo de regressão logística
    const logistic = new LogisticRegression({
        numSteps: 10000,      // Número de iterações aumentadas
        learningRate: 1e-4,   // Taxa de aprendizado ajustada
        lambda: 0.01          // Regularização ajustada
    });

    // Treinar o modelo com os dados
    logistic.train(X_train, Y_train);
    console.log('Modelo treinado com sucesso.');

    // Fazer previsões com os dados de teste
    const predictions = logistic.predict(X_test);

    // Exibir as previsões e rótulos verdadeiros
    console.log('Previsões:', predictions);
    console.log('Rótulos verdadeiros:', Y_test.to1DArray());

    // Calcular e exibir a acurácia
    const accuracy = calculateAccuracy(predictions, Y_test.to1DArray());
    console.log('Acurácia do modelo:', accuracy);
}

// Função para calcular a acurácia baseada em uma margem de erro de 5%
function calculateAccuracy(predictions, trueLabels) {
    let correctCount = 0;
    const threshold = 0.05; // Definimos uma margem de 5% de erro

    for (let i = 0; i < predictions.length; i++) {
        const predictedValue = predictions[i];
        const actualValue = trueLabels[i];

        // Verifica se a previsão está dentro de 5% do valor real
        if (Math.abs(predictedValue - actualValue) / actualValue <= threshold) {
            correctCount++;
        }
    }

    // Acurácia é o número de previsões corretas dividido pelo total de previsões
    return (correctCount / predictions.length) * 100;
}

function normalizeData(features) {
    // Normalização Z-score: (x - média) / desvio padrão

    const meanAdjClose = features.reduce((acc, val) => acc + val[0], 0) / features.length;
    const meanClose = features.reduce((acc, val) => acc + val[1], 0) / features.length;

    const stdAdjClose = Math.sqrt(features.reduce((acc, val) => acc + Math.pow(val[0] - meanAdjClose, 2), 0) / features.length);
    const stdClose = Math.sqrt(features.reduce((acc, val) => acc + Math.pow(val[1] - meanClose, 2), 0) / features.length);

    // Normaliza cada feature
    features.forEach((val, idx) => {
        features[idx] = [
            (val[0] - meanAdjClose) / stdAdjClose,  // Normaliza Adj_Close
            (val[1] - meanClose) / stdClose        // Normaliza Close
        ];
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
