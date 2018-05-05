from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import socket                                         
import time
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

lines = open('output.tsv', encoding='utf-8').read().split('\n')

que_sentences = []
ans_sentences = []
alphabets = ['q','p','o','i','u','y','t','r','e','w','l','k','j','h','g','f','d','s','a','z','x','c','v','b','n','m','0','1','2','3','4','5','6','7','8','9']
que_word = set()
ans_word = set()
nb_samples = 3000


for line in range(nb_samples):
    
    que_line = str(lines[line]).split('\t')[0]
    ans_line = '\t' + str(lines[line]).split('\t')[1] + '\n'
    que_sentences.append(str(que_line).lower())
    ans_sentences.append(str(ans_line).lower())
    que_wrd = que_line.lower().split(' ')
    for wrd in que_wrd:
        c = 0
        for char in range(0,len(wrd)):
            if(wrd[char] in alphabets):
                c=c+1
            else:
                break
        que_word.add(wrd[0:c])
    ans_wrd = ans_line.lower().split(' ')
    for wrd in ans_wrd:
        c = 0
        for char in range(0,len(wrd)):
            if(wrd[char] in alphabets):
                c=c+1
            else:
                break
        ans_word.add(wrd[0:c])
que_word.add(' ')
ans_word.add(' ')
que_word = sorted(list(que_word))
ans_word = sorted(list(ans_word))
que_word.pop(0)
ans_word.pop(0)

que_index_to_word_dict = {}
que_word_to_index_dict = {}
for k, v in enumerate(que_word):
    que_index_to_word_dict[k] = v
    que_word_to_index_dict[v] = k
        

ans_index_to_word_dict = {}
ans_word_to_index_dict = {}
for k, v in enumerate(ans_word):
    ans_index_to_word_dict[k] = v
    ans_word_to_index_dict[v] = k


max_len_que_sent = max([len(line) for line in que_sentences])
max_len_ans_sent = max([len(line) for line in ans_sentences])

tokenized_que_sentences = np.zeros(shape = (nb_samples,max_len_que_sent,len(que_word)), dtype='float32')
tokenized_ans_sentences = np.zeros(shape = (nb_samples,max_len_ans_sent,len(ans_word)), dtype='float32')
target_data = np.zeros((nb_samples, max_len_ans_sent, len(ans_word)),dtype='float32')

for i in range(nb_samples):
    a = que_sentences[i].lower().split(' ')
    """b=[]
    j=len(a)*2-1
    k=0
    for i in range(0,j):
        if(i%2==0):
            b.append(a[k])
            k+=1
        else:
            b.append(' ')
            
    a=b"""    
    for k in range(0,len(a)):
        c=0
        for char in range(0,len(a[k])):
            if(a[k][char] in alphabets):
                c=c+1
            else:
                break
        #print(a[k][0:c])
        s = a[k][0:c]
        #print(s)
        #print("hello")
        if(s!=''):
            tokenized_que_sentences[i,k,que_word_to_index_dict[s]] = 1.

    a = ans_sentences[i].lower().split(' ')
    """j=len(a)*2-1
    k=0
    b=[]
    for i in range(0,j):
        if(i%2==0):
            b.append(a[k])
            k+=1
        else:
            b.append(' ')
            
    a=b"""
    for k in range(0,len(a)):
        c=0
        for char in range(0,len(a[k])):
            if(a[k][char] in alphabets):
                c=c+1
            else:
                break
        s = a[k][0:c]
        if(s!=''):
            #print(s)
            tokenized_ans_sentences[i,k,ans_word_to_index_dict[s]] = 1.   

        if k > 0:
            if(s!=''):
                target_data[i,k-1,ans_word_to_index_dict[s]] = 1.

encoder_input = Input(shape=(None,len(que_word)))
encoder_LSTM = LSTM(256,return_state = True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM (encoder_input)
encoder_states = [encoder_h, encoder_c]

decoder_input = Input(shape=(None,len(ans_word)))
decoder_LSTM = LSTM(256,return_sequences=True, return_state = True)
decoder_out, _ , _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(ans_word),activation='softmax')
decoder_out = decoder_dense (decoder_out)

model = Model(inputs=[encoder_input, decoder_input],outputs=[decoder_out])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[tokenized_que_sentences,tokenized_ans_sentences], 
          y=target_data,
          batch_size=64,
          epochs=1000,
          validation_split=0.2)
model.save('s2sword.h5')

encoder_model_inf = Model(encoder_input, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, 
                                                 initial_state=decoder_input_states)

decoder_states = [decoder_h , decoder_c]

decoder_out = decoder_dense(decoder_out)

decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states )

def decode_seq(inp_seq): 
    states_val = encoder_model_inf.predict(inp_seq)    
    target_seq = np.zeros((1, 1, len(ans_word)))
    target_seq[0, 0,0] = 1.
    
    translated_sent = ''
    stop_condition = False
    
    while not stop_condition:
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[target_seq] + states_val)
        max_val_index = np.argmax(decoder_out[0,-1,:])
        sampled_ans_char = ans_index_to_word_dict[max_val_index]
        translated_sent += sampled_ans_char
        translated_sent += ' '
        
        if ( (sampled_ans_char == '\n') or (len(translated_sent) > max_len_ans_sent)) :
            stop_condition = True
        
        target_seq = np.zeros((1, 1, len(ans_word)))
        target_seq[0, 0, max_val_index] = 1
        states_val = [decoder_h, decoder_c]
    return translated_sent

'''f = open("output2word.txt", 'w+')
for seq_index in range(10):
    inp_seq = tokenized_que_sentences[seq_index:seq_index+1]
    translated_sent = decode_seq(inp_seq)
    print('-')
    print('Input sentence:', que_sentences[seq_index])
    print(len(translated_sent))
    print('Decoded sentence:', translated_sent)
    f.write(str(input_texts[seq_index])+','+str(decoded_sentence))
f.close()'''

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = "0.0.0.0"
port = 9000
serversocket.bind((host,port))
serversocket.listen(5)
while True:
    # establish a connection
    print('Listening')
    clientsocket,addr = serversocket.accept()
    print("Connection from: " + str(addr))
    z = clientsocket.recv(20240)
    x = z.decode('utf8')
    for t, char in enumerate(x):
        encoder_input_data[0, t, input_token_index[char]] = 1.

    input_seq = encoder_input_data[0:0+1]
    decoded_sentence = decode_sequence(input_seq)
    print("Answer is "+decoded_sentence)

    y = decoded_sentence
    clientsocket.send(y.encode('ascii'))
    clientsocket.close()
