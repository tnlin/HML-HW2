from data_utils_cora import *
import pandas as pd
from gensim.models import Word2Vec

class DeepWalk():
    def __init__(self, A):
        self.G = nx.DiGraph(A, nodetype=int)
        self.G = read_cora('data/cora/cora.cites', directed=True)

    def walk(self, n_step, start_node):
        walk = [start_node]

        while len(walk) < n_step:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(np.random.choice(cur_nbrs))
            else:
                break
        return walk

    def train(self, n_iter, n_step):
        walks = []
        nodes = list(self.G.nodes())
        print('Walking for %d iteration (step=%d)' % (n_iter, n_step))
        for walk_iter in range(n_iter):
            print(str(walk_iter+1), end=' ')
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(n_step, node))
        return walks
    

if __name__=='__main__':
    X, A, y = load_data(dataset='cora')
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y)

    model = DeepWalk(A)
    walks = model.train(100, 100)
    walks = [[str(x) for x in walk] for walk in walks]

    print("\nBuilding embedding...")
    model = Word2Vec(walks, size=256, sg=1, workers=4)
    model.wv.save_word2vec_format('cora.emb')

    print("Evaluating...")
    embedding = pd.read_csv('cora.emb',sep=' ',skiprows=1,header = None, index_col=0)
    label = pd.read_csv('data/cora/cora.content',sep='	',header = None, index_col=0)
    label = label.iloc[:,-1]
    train_idx, test_idx = label.iloc[idx_test].index, label.iloc[idx_test].index
    train_label, test_label = label.loc[train_idx], label.loc[test_idx]
    train_embedding, test_embedding = embedding.loc[train_idx], embedding.loc[test_idx]

    lr, accuracy = evaluate_cora(train_embedding.values, test_embedding.values, train_label, test_label)
    print("Accuracy:", accuracy)
    