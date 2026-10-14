from FlagEmbedding import BGEM3FlagModel

model_path = 'model/BAAI/bge-m3'
model = BGEM3FlagModel(model_path)

sentences = ["That is a happy person","That is a happy dog","That is a very happy person","Today is a sunny day"]
all_vectors = model.encode(sentences,return_dense=True,return_sparse=True)

dense_vectors = all_vectors["dense_vecs"]
sparse_vectors = all_vectors['lexical_weights']

print(dense_vectors.shape)
print(len(sparse_vectors))

def hello():
    print("hello")

def tool5():
    print("tool5")

def tool7():
    print("tool7")
    
def tool6():
    print("tool6")
    

