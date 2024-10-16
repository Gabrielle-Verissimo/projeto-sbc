import numpy as np

#palavras
def generateTable(data, k):
    T = {}
    for i in range(len(data)-k):
        X = tuple(data[i:i+k])
        Y = data[i+k]
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T

def convertFreqIntoProb(T):     
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
                
    return T

def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text_path = "literatura.txt"
text = load_text(text_path)

def MarkovChain(text,k):
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    return T

model = MarkovChain(text, 10)

#palavra
def sample_next(ctx, model, k):
    ctx = tuple(ctx[-k:])
    if model.get(ctx) is None:
        aux = 'literatura'
        ctx = tuple(aux[-k:])
    
    possible_words = list(model[ctx].keys())
    possible_values = list(model[ctx].values())
    
    return np.random.choice(possible_words, p=possible_values)

def generateText(starting_sent,k,maxLen=2000):
    sentence = starting_sent
    ctx = starting_sent[-k:]

    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    while sentence[-1] not in [".", "!", "?"]:
        next_word = sample_next(ctx, model, k)
        sentence += next_word
        ctx = sentence[-k:]

    return sentence
text = generateText("literatura",k=10,maxLen=500)
print(text)