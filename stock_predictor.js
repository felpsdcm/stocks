const LogisticRegression = require('ml-logistic-regression');
const csv = require('csv-parser');
const fs = require('fs');
const { Matrix } = require('ml-matrix');

async function trainModel() {
    const features = [];
    const labels = [];
    const maxRows = 100;  // Limite de 100 linhas para amostra

    // Ler o arquivo CSV com limite de linhas
    await new Promise((resolve, reject) => {
        let rowCount = 0;  // Contador de linhas lidas
        let lastClose = null;  // Para armazenar o preço de fechamento do dia anterior

        fs.createReadStream('GoogleData.csv')  // Substitua pelo seu arquivo CSV real
            .pipe(csv())
            .on('data', (row) => {
                if (rowCount >= maxRows) return;

                const adjClose = parseFloat(row.Adj_Close);
                const close = parseFloat(row.Close);

                if (!isNaN(adjClose) && !isNaN(close) && lastClose !== null) {
                    // Definir o rótulo: se o preço de fechamento de hoje for maior que o de ontem, é 1 (subiu), caso contrário, 0 (desceu)
                    const label = close > lastClose ? 1 : 0;
                    features.push([adjClose, close]);  // Usar adjClose e close como características
                    labels.push(label);  // Rótulo binário de "subiu" ou "desceu"
                }

                lastClose = close;  // Atualiza o preço de fechamento do último dia
                rowCount++;
            })
            .on('end', () => {
                resolve();  // Resolve a promise quando terminar a leitura
            })
            .on('error', reject);  // Se ocorrer erro, rejeita a promise
    });

    console.log('Dados de treinamento lidos com sucesso.');

    // Normalizar os dados
    normalizeData(features);

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
    console.log('Previsões (0 = desceu, 1 = subiu):', predictions);
    console.log('Rótulos verdadeiros (0 = desceu, 1 = subiu):', Y_test.to1DArray());

    // Calcular e exibir a acurácia
    const accuracy = calculateAccuracy(predictions, Y_test.to1DArray());
    console.log('Acurácia do modelo:', accuracy);

    // Exibir mensagens baseadas nas previsões
    predictions.forEach((prediction, index) => {
        if (prediction === 0) {
            console.log(`Previsão ${index + 1}: Não é adequado investir nesta ação. A ação está em queda.`);
        } else {
            console.log(`Previsão ${index + 1}: Excelente! A ação está em alta, pode ser um bom momento para investir.`);
        }
    });
}

// Função para calcular a acurácia
function calculateAccuracy(predictions, trueLabels) {
    let correctCount = 0;
    for (let i = 0; i < predictions.length; i++) {
        if (predictions[i] === trueLabels[i]) {
            correctCount++;
        }
    }
    return (correctCount / predictions.length) * 100;
}

// Função para normalizar os dados (Z-score)
function normalizeData(features) {
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
