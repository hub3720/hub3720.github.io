const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const fetch = require('node-fetch');
const app = express();
const PORT = 3000;

app.use(cors());
app.use(bodyParser.json());

/* --- SIMULAÇÃO DE NEURÔNIOS --- */
class Neuron {
    constructor(id){
        this.id = id;
        this.weights = [];
        this.bias = Math.random()*0.2-0.1;
        this.value = 0;
        this.connections = [];
    }
    connect(neuron){
        const w = Math.random()*0.2-0.1;
        this.connections.push({neuron:neuron,weight:w});
        this.weights.push(w);
    }
    activate(input){
        let sum = input+this.bias;
        this.value = 1/(1+Math.exp(-sum)); // Sigmoid
        return this.value;
    }
}

/* --- CRIAÇÃO DE REDE NEURAL --- */
const NUM_INPUT = 50;
const NUM_HIDDEN = 30000; // 30 mil neurônios
const NUM_OUTPUT = 50;

let inputLayer = [];
for(let i=0;i<NUM_INPUT;i++) inputLayer.push(new Neuron(i));
let hiddenLayer = [];
for(let i=0;i<NUM_HIDDEN;i++) hiddenLayer.push(new Neuron(i));
let outputLayer = [];
for(let i=0;i<NUM_OUTPUT;i++) outputLayer.push(new Neuron(i));

// conexões input -> hidden
for(let n of inputLayer){
    for(let h of hiddenLayer){
        n.connect(h);
    }
}
// conexões hidden -> output
for(let h of hiddenLayer){
    for(let o of outputLayer){
        h.connect(o);
    }
}

/* --- MEMÓRIA DA IA --- */
let memory = []; // aprendizado online simples

/* --- FUNÇÃO DE BUSCA SIMULADA --- */
async function searchWeb(query){
    // simulando 10 fontes
    let results = [];
    for(let i=0;i<10;i++){
        results.push(`Informação ${i+1} sobre "${query}" coletada da web.`);
    }
    return results;
}

/* --- INTERPRETAÇÃO DAS FONTES --- */
function interpretSources(sources){
    // simples consolidação, mistura textos
    let combined = sources.join(' ');
    // simplificação de linguagem
    combined = combined.replace(/informação \d+ sobre/g,'dados sobre');
    return combined;
}

/* --- PROCESSAMENTO DA REDE NEURAL --- */
function neuralProcess(query){
    let inputVector = [];
    for(let i=0;i<NUM_INPUT;i++){
        inputVector.push(Math.random());
    }
    for(let i=0;i<NUM_INPUT;i++){
        inputLayer[i].activate(inputVector[i]);
    }
    // hidden layer
    for(let h of hiddenLayer){
        let sum=0;
        for(let c of h.connections){
            sum+=c.neuron.value*c.weight;
        }
        h.activate(sum);
    }
    // output layer
    let outputValues = [];
    for(let o of outputLayer){
        let sum=0;
        for(let c of o.connections){
            sum+=c.neuron.value*c.weight;
        }
        outputValues.push(o.activate(sum));
    }
    // traduz para resposta simulada
    let answer = outputValues.slice(0,5).map(v=>Math.floor(v*100)).join(' ')+" — resposta gerada neuralmente";
    return answer;
}

/* --- APRENDIZADO ONLINE --- */
function learn(query, answer){
    memory.push({query,answer});
    if(memory.length>500) memory.shift();
}

/* --- ROTA PRINCIPAL --- */
app.post('/query',async (req,res)=>{
    const query = req.body.query || '';
    if(!query) return res.json({answer:'Pergunta vazia'});

    try{
        const sources = await searchWeb(query);
        const interpreted = interpretSources(sources);
        const neuralAnswer = neuralProcess(query);

        const finalAnswer = `${interpreted} \n\n[IA Processada]: ${neuralAnswer}`;

        learn(query,finalAnswer);
        res.json({answer:finalAnswer});
    }catch(err){
        console.error(err);
        res.json({answer:'Erro ao processar a pergunta'});
    }
});

/* --- ROTA MEMÓRIA --- */
app.get('/memory',(req,res)=>{
    res.json({memory});
});

/* --- ROTA LIMPAR MEMÓRIA --- */
app.post('/memory/clear',(req,res)=>{
    memory = [];
    res.json({status:'Memória limpa'});
});

/* --- LOGS SIMULADOS --- */
setInterval(()=>{
    console.log(`Memória atual: ${memory.length} entradas. Processando ${Math.floor(Math.random()*1000)} neurônios.`);
},5000);

/* --- ROTA TESTE --- */
app.get('/',(req,res)=>{
    res.send('Servidor IA rodando. Use /query com POST para interagir.');
});

app.listen(PORT,()=>{console.log(`Servidor rodando em http://localhost:${PORT}`);});
