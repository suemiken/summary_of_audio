from common.np import *
class EM_TF_IDF_Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx, tf_idf, ems, train):
        W, = self.params
        self.idx = idx
        
        if train:
            f_tf_idf = []
            
            # tf_idfの整形3*1
            for (id, one_td) in zip(idx, tf_idf):
                taihi_a = []
                taihi_a.append(one_td[id])
                f_tf_idf.append(taihi_a)

            tf_idf = np.array(f_tf_idf)
            
            #emの整形3*1
            ems = np.array([ems]).T
            
            # print(tf_idf)
            out = W[idx] * tf_idf * ems
        else:
            out = W[idx]
        
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None

class EM_TF_IDF_TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs, tf_idf, em, word_to_id ,train):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []
        
        #単語に割り当てる重みの計算
        if train:
            i = 0
            ems = []
            for (ea, xa) in zip(em, xs):
                e = []
                for x in xa:
                    if x == word_to_id['。']:
                        i = i + 1                
                        e.append(0.1)
                    elif x == word_to_id['null']:
                        e.append(0.1)
                    else:               
                        e.append(ea[i])    
                ems.append(e)
            em = np.array(ems)
            
            for t in range(T):
                layer = EM_TF_IDF_Embedding(self.W)
                out[:, t, :] = layer.forward(xs[:, t], tf_idf, em[:, t], train)
                self.layers.append(layer)
                
        else:
            for t in range(T):
                layer = EM_TF_IDF_Embedding(self.W)
                out[:, t, :] = layer.forward(xs[:, t], tf_idf, em, train)
                self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None
