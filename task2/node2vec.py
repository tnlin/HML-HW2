from data_utils_cora import *
import pandas as pd
from gensim.models import Word2Vec


class Noed2Vec():
    def __init__(self, directed, p, q):
        self.G = read_cora('data/cora/cora.cites', directed=True)
        self.directed = directed
        self.p = p
        self.q = q
        self.preprocess_transition_probs()

    def walk(self, n_step, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < n_step:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def train(self, n_iter, n_step):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walking for %d iteration (step=%d)' % (n_iter, n_step))
        for walk_iter in range(n_iter):
            print(str(walk_iter+1), end=' ')
            np.random.shuffle(nodes)
            for node in nodes:
                walks.append(self.walk(n_step=n_step, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        directed = self.directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        return
    
    def alias_setup(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q
    
    def alias_draw(self, J, q):
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
        
        
if __name__=='__main__':
    X, A, y = load_data(dataset='cora')
    y_train, y_val, y_test, idx_train, idx_val, idx_test = get_splits(y)

    model = Noed2Vec(directed=True, p=1, q=1)
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