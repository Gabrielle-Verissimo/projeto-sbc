def generateTable(data,k=2):
    
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

def convertFreqIntoProb(T):     
    for kx in T.keys():
        s = float(sum(T[kx].values()))
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
                
    return T
 


text_path = "literatura.txt"
def load_text(filename):
    with open(filename,encoding='utf8') as f:
        return f.read().lower()
    
text = load_text(text_path)

def MarkovChain(text,k=2):
    T = generateTable(text,k)
    T = convertFreqIntoProb(T)
    return T
 
model = MarkovChain(text)

import numpy as np

def sample_next(ctx,model,k):
 
    ctx = ctx[-k:]
    if model.get(ctx) is None:
        return " "
    possible_Chars = list(model[ctx].keys())
    possible_values = list(model[ctx].values())
    
 
    return np.random.choice(possible_Chars,p=possible_values)
 

import numpy as np

def generateText(starting_sent,k=2,maxLen=1000):
    
    sentence = starting_sent
    ctx = starting_sent[-k:]
    
    for ix in range(maxLen):
        next_prediction = sample_next(ctx,model,k)
        sentence += next_prediction
        ctx = sentence[-k:]
    return sentence
 
 
text = generateText("litiratura",k=2,maxLen=500)
print(text)