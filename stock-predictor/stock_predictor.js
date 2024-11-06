// Importando a biblioteca corretamente
const LogisticRegression = require('ml-logistic-regression').LogisticRegression;
const { Matrix } = require('ml-matrix');
const csv = require('csv-parser');
const fs = require('fs');

// Função para carregar os dados do CSV
function loadCSVData(filePath) {
    return new Promise((resolve, reject) => {
        const data = [];
        fs.createReadStream(filePath)
            .pipe(csv())
            .on('data', (row) => {
                data.push(row);
            })
            .on('end', () => {
                resolve(data);
            })
            .on('error', (error) => reject(error));
    });
}

// Função para preparar os dados
function prepareData(rawData) {
    const X = [];
    const y = [];

    for (let i = 0; i < rawData.length - 1; i++) {
        // Características: Open, High, Low, Volume
        const row = [
            parseFloat(rawData[i]['Open']),
            parseFloat(rawData[i]['High']),
            parseFloat(rawData[i]['Low']),
            parseFloat(rawData[i]['Volume']),
        ];
        X.push(row);

        // Alvo: 1 se o próximo dia Close for maior do que o Close do dia atual
        const target = rawData[i + 1]['Close'] > rawData[i]['Close'] ? 1 : 0;
        y.push(target);
    }

    return { X: new Matrix(X), y };
}

// Função principal para treinar o modelo
async function trainModel() {
    // Carregar os dados do CSV
    const data = await loadCSVData('microsoft.csv'); // Certifique-se de que o arquivo CSV esteja no diretório correto

    // Preparar os dados
    const { X, y } = prepareData(data);
    
    // Criar o modelo de regressão logística
    const logreg = new LogisticRegression({
        numSteps: 1000,
        learningRate: 5e-3,
        batchSize: 10,
    });

    // Treinar o modelo
    logreg.train(X, y);

    // Fazer previsões
    const predictions = logreg.predict(X);

    // Avaliar a acurácia
    const accuracy = calculateAccuracy(predictions, y);
    console.log(`Acurácia do modelo: ${(accuracy * 100).toFixed(2)}%`);
}

// Função para calcular a acurácia
function calculateAccuracy(predictions, y) {
    let correct = 0;
    for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === y[i]) {
            correct++;
        }
    }
    return correct / predictions.length;
}

// Chamar a função principal
trainModel().catch((error) => console.error('Erro ao treinar o modelo:', error));
