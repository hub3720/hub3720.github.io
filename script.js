// ====================================================================
// PARTE 1: IMPLEMENTAÇÃO DA REDE NEURAL ARTIFICIAL (RNA)
// ====================================================================

/**
 * Funções de Ativação
 * Usaremos a função ReLU para as camadas ocultas e Softmax (simplificada) 
 * na camada de saída para obter probabilidades.
 */
const activationFunctions = {
    // Retificador Linear Unidade (ReLU)
    relu: (x) => Math.max(0, x),
    
    // Simplificação da Softmax: Apenas a função de expoente.
    // O array de saída será normalizado manualmente depois.
    exp: (x) => Math.exp(x)
};

/**
 * Função de Forward Propagation
 * Calcula a saída de uma camada (Layer)
 * Saída = f_ativacao( (Entrada * Pesos) + Bias )
 * * @param {number[]} input - O vetor de entrada da camada (saída da camada anterior).
 * @param {number[][]} weights - A matriz de pesos (input_size x output_size).
 * @param {number[]} biases - O vetor de bias (output_size).
 * @param {string} activation - A chave da função de ativação a ser usada.
 * @returns {number[]} O vetor de saída da camada.
 */
function forwardPropagate(input, weights, biases, activation) {
    const activationFn = activationFunctions[activation];
    const outputSize = biases.length;
    const inputSize = input.length;
    const output = new Array(outputSize).fill(0);

    // Itera sobre cada neurônio na camada de saída
    for (let j = 0; j < outputSize; j++) {
        let sum = 0;
        
        // Calcula o produto escalar (dot product) da entrada com os pesos sinápticos
        // Sum = Sum(Input[i] * Weights[i][j])
        for (let i = 0; i < inputSize; i++) {
            sum += input[i] * weights[i][j];
        }
        
        // Adiciona o bias
        sum += biases[j];
        
        // Aplica a função de ativação
        output[j] = activationFn(sum);
    }
    
    return output;
}

/**
 * Função para gerar um número aleatório entre min e max
 */
function randomRange(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Função para gerar pesos e bias aleatórios para simular o treinamento.
 * O preenchimento garante que a rede de 3105 neurônios seja criada.
 */
function initializeWeights(inputSize, outputSize) {
    const weights = [];
    for (let i = 0; i < inputSize; i++) {
        const row = [];
        for (let j = 0; j < outputSize; j++) {
            // Inicialização de Xavier/Glorot (simplificada)
            row.push(randomRange(-0.5, 0.5)); 
        }
        weights.push(row);
    }
    const biases = new Array(outputSize).fill(0).map(() => randomRange(-0.5, 0.5));
    return { weights, biases };
}


// ====================================================================
// PARTE 2: PRÉ-PROCESSAMENTO, CARREGAMENTO DE DADOS E LÓGICA DO CHAT
// ====================================================================

let modelData = {};
const $chatContainer = document.getElementById('chat-container');
const $userInput = document.getElementById('user-input');
const $sendBtn = document.getElementById('send-btn');

/**
 * Carrega os dados do modelo (vocabulário, intents, pesos)
 */
async function loadModelData() {
    try {
        const response = await fetch('model_data.json');
        modelData = await response.json();

        // **GARANTINDO A ESTRUTURA DA REDE DE 3105 NEURÔNIOS**
        const vocabSize = 100; // 100 neurônios de entrada
        const h1Size = 1500;   // 1500 neurônios ocultos 1
        const h2Size = 1500;   // 1500 neurônios ocultos 2
        const outputSize = 5;  // 5 neurônios de saída (categorias)

        // Se os dados estiverem vazios, inicializa a estrutura da rede com pesos aleatórios
        if (modelData.weights_h1.length === 0) {
            console.log("Inicializando pesos aleatórios para simular uma rede treinada...");
            
            let w_b;
            
            // Input (100) -> Hidden 1 (1500)
            w_b = initializeWeights(vocabSize, h1Size);
            modelData.weights_h1 = w_b.weights;
            modelData.bias_h1 = w_b.biases;
            
            // Hidden 1 (1500) -> Hidden 2 (1500)
            w_b = initializeWeights(h1Size, h2Size);
            modelData.weights_h2 = w_b.weights;
            modelData.bias_h2 = w_b.biases;

            // Hidden 2 (1500) -> Output (5)
            w_b = initializeWeights(h2Size, outputSize);
            modelData.weights_out = w_b.weights;
            modelData.bias_out = w_b.biases;

            console.log(`Estrutura da rede com ${vocabSize + h1Size + h2Size + outputSize} neurônios criada.`);
        }

    } catch (error) {
        console.error("Erro ao carregar dados do modelo:", error);
        $chatContainer.innerHTML += '<div class="message bot"><span class="text">Erro ao carregar o modelo. Verifique o arquivo model_data.json.</span></div>';
    }
}

/**
 * Converte a frase de entrada para um vetor Bag-of-Words.
 * @param {string} sentence - Frase do usuário.
 * @returns {number[]} Vetor de entrada da RNA.
 */
function sentenceToBagOfWords(sentence) {
    // 1. Tokenização e Normalização
    const tokens = sentence.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
    const vocab = modelData.vocabulary;
    const bag = new Array(vocab.length).fill(0);

    // 2. Preenchimento do vetor BoW
    tokens.forEach(word => {
        const index = vocab.indexOf(word);
        if (index !== -1) {
            bag[index] = 1; // 1 se a palavra existe, 0 se não
        }
    });

    return bag;
}

/**
 * Normaliza um vetor de saída usando o Softmax
 * Softmax(z_i) = e^(z_i) / Sum(e^(z_j))
 * @param {number[]} outputVector - Saída bruta (pós-ativação exp) da camada de saída.
 * @returns {number[]} Vetor de probabilidades normalizadas.
 */
function softmax(outputVector) {
    const sumExp = outputVector.reduce((acc, val) => acc + val, 0);
    
    // Evita divisão por zero. Se a soma for zero, retorna um vetor de zeros.
    if (sumExp === 0) return new Array(outputVector.length).fill(0);
    
    return outputVector.map(val => val / sumExp);
}


/**
 * Executa a Forward Propagation completa na rede
 * @param {number[]} inputVector - Vetor BoW de 100 elementos.
 * @returns {number[]} Vetor de probabilidade (5 elementos).
 */
function runNeuralNetwork(inputVector) {
    // 1. Input (100) -> Hidden 1 (1500)
    let h1_output = forwardPropagate(
        inputVector, 
        modelData.weights_h1, 
        modelData.bias_h1, 
        'relu'
    );

    // 2. Hidden 1 (1500) -> Hidden 2 (1500)
    let h2_output = forwardPropagate(
        h1_output, 
        modelData.weights_h2, 
        modelData.bias_h2, 
        'relu'
    );

    // 3. Hidden 2 (1500) -> Output (5) - Usa 'exp' para o numerador da Softmax
    let output_raw = forwardPropagate(
        h2_output, 
        modelData.weights_out, 
        modelData.bias_out, 
        'exp' // Usa a função exp() como pré-Softmax
    );

    // 4. Normalização Softmax final para obter probabilidades
    let output_probs = softmax(output_raw);

    return output_probs;
}

/**
 * Processa a entrada do usuário e gera a resposta do bot.
 */
function getBotResponse(userMessage) {
    if (!userMessage.trim()) return "Por favor, digite uma mensagem válida.";
    
    // 1. Pré-processamento: Cria o vetor de entrada
    const inputVector = sentenceToBagOfWords(userMessage);

    // 2. Executa a RNA (Forward Propagation)
    const outputProbs = runNeuralNetwork(inputVector);
    
    // Encontra a intenção (tag) com a maior probabilidade
    let maxProb = -1;
    let bestTagIndex = -1;
    outputProbs.forEach((prob, index) => {
        if (prob > maxProb) {
            maxProb = prob;
            bestTagIndex = index;
        }
    });

    const predictedTag = modelData.tags[bestTagIndex];
    const threshold = 0.7; // Limiar de confiança

    console.log(`Probabilidades: [${outputProbs.map(p => p.toFixed(2)).join(', ')}]`);
    console.log(`Previsão: ${predictedTag} com Confiança: ${maxProb.toFixed(4)}`);
    
    // 3. Geração da Resposta
    if (maxProb > threshold) {
        // Se a confiança for alta, seleciona uma resposta da intenção prevista
        const intent = modelData.intents.find(i => i.tag === predictedTag);
        if (intent && intent.responses.length > 0) {
            const randomIndex = Math.floor(Math.random() * intent.responses.length);
            return intent.responses[randomIndex];
        }
    }
    
    // Resposta padrão (fallback)
    return "Desculpe, não entendi. Minha rede neural ainda é simples!";
}

/**
 * Adiciona uma nova mensagem ao container do chat.
 */
function appendMessage(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);
    messageDiv.innerHTML = `<span class="text">${message}</span>`;
    $chatContainer.appendChild(messageDiv);
    
    // Scroll para a última mensagem
    $chatContainer.scrollTop = $chatContainer.scrollHeight;
}

/**
 * Lida com o envio da mensagem
 */
function handleSendMessage() {
    const userMessage = $userInput.value.trim();
    if (userMessage === "") return;

    // 1. Exibe a mensagem do usuário
    appendMessage(userMessage, 'user');
    
    // 2. Limpa o input
    $userInput.value = '';

    // 3. Obtém e exibe a resposta do bot
    setTimeout(() => {
        const botResponse = getBotResponse(userMessage);
        appendMessage(botResponse, 'bot');
    }, 500); // Pequeno delay para simular o processamento
}


document.addEventListener('DOMContentLoaded', () => {
    // 1. Carrega a estrutura da RNA
    loadModelData();

    // 2. Listener para o botão de envio
    $sendBtn.addEventListener('click', handleSendMessage);
    
    // 3. Listener para a tecla Enter no campo de texto
    $userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });
});
