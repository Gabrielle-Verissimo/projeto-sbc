import numpy as np

#Caracter
def generateTable(data,k):
    
    T = {}
    for i in range(len(data)-k):
        X = data[i:i+k]
        Y = data[i+k]
        #print("X  %s and Y %s  "%(X,Y))
        
        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1
    
    return T

# def convertFreqIntoProb(T):     
#     for kx in T.keys():
#         s = float(sum(T[kx].values()))
#         for k in T[kx].keys():
#             T[kx][k] = T[kx][k]/s
                
#     return T

def convertFreqIntoProb_Laplace(T, vocab_size, alpha=1):
    for kx in T.keys():
        total_count = float(sum(T[kx].values()) + alpha * vocab_size)  # soma das frequências + suavização
        for k in T[kx].keys():
            T[kx][k] = (T[kx][k] + alpha) / total_count  # aplicando suavização de Laplace
    
    return T

def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text_path = "teste2.txt"
text = load_text(text_path)

# def MarkovChain(text,k):
#     T = generateTable(text,k)
#     T = convertFreqIntoProb(T)
#     return T

def MarkovChain(text, k, alpha=1):
    vocab_size = len(set(text))  # calcular o tamanho do vocabulário
    T = generateTable(text, k)
    T = convertFreqIntoProb_Laplace(T, vocab_size, alpha)
    return T
 
model = MarkovChain(text, 6)

#Caracter
def sample_next(ctx,model,k):
 
    ctx = ctx[-k:]
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())
    #max_index = np.argmax(possible_values)
    
    #print(possible_Chars)
    #print(possible_values)
    total_prob = sum(possible_values)
    if total_prob != 1:
        possible_values = [p / total_prob for p in possible_values]
 
    return np.random.choice(possible_Chars,p=possible_values)

def generateText(starting_sent,k,maxLen=2000):
    
    sentence = starting_sent
    ctx = starting_sent[-k:]
    
    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    while sentence[-1] not in [".", "!", "?"]:
        next_prediction = sample_next(ctx, model, k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence
 
text = generateText("Jane",k=6,maxLen=500)
print(text)