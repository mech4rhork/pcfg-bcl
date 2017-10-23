
## pcfg_bcl.py

import re
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.grammar import PCFG, Nonterminal, ProbabilisticProduction
from nltk import bigrams, sent_tokenize, word_tokenize
from coclust.CoclustInfo import CoclustInfo

#
#   VARIABLES
#------------------------------------

and_symb_count = -1
or_symb_count = -1
biclusters = {}
ignore_mc_ec = False

ALPHA = 1. # 1.0
LPG_DIFF_THRESHOLD = 100. # 1.0
MC_THRESHOLD = 10. # 2.0

#
#   FONCTIONS OUTILS
#------------------------------------

def _get_and_symb_index():
    global and_symb_count
    and_symb_count += 1
    return and_symb_count
    
def _get_or_symb_index():
    global or_symb_count
    or_symb_count += 1
    return or_symb_count

def _create_t(C):
    text = C # corpus
    text_wp = re.sub(r'[^\w\s]', '', text) # sans ponctuation
    ## bigrammes
    bigram_count = {} # contient le nombre de fois que 2 mots forment une pair
    for sentence in sent_tokenize(text):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = word_tokenize(sentence)
        pairs = [" ".join(pair) for pair in bigrams(tokens)]
        for b in pairs:
            if b not in bigram_count:
                bigram_count[b] = 1
            else:
                bigram_count[b] += 1
    ## vocabulaire: indice unique pour chaque mot
    vocabulary = defaultdict()
    vocabulary.default_factory = lambda: len(vocabulary)
    [vocabulary[word] for word in text_wp.split()]
    voc_index = [(vocabulary[v], v) for v in vocabulary] # liste de (indice,'mot')
    voc_index = sorted(voc_index, key=lambda x: x[0]) # triée selon l'indice
    ## matrice mot-mot (T)
    n = len(vocabulary)
    T = np.zeros((n,n), dtype=int)
    for bigram in bigram_count:
        b = bigram.split()
        #print(b[0], b[1])
        T[vocabulary[b[0]],vocabulary[b[1]]] += bigram_count[bigram]
    t_names = [e[1] for e in voc_index]
    T_df = pd.DataFrame(T, index=t_names, columns=t_names)
    ## suppression des lignes et colonnes nulles
    T_df = T_df.loc[(T_df.sum(axis=1) != 0), (T_df.sum(axis=0) != 0)]
    return T_df

def _create_ec(BC, C, T):
    ## expressions
    expressions = []
    for i in BC.index:
        for j in BC.columns:
            expressions.append((i,j))
    ## contextes
    ec_dict = {}
    for expr in expressions:
        ec_dict[expr] = {}
    contexts = []
    for sentence in sent_tokenize(C):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        tokens = [-1] + word_tokenize(sentence) + [-1]
        ## contextes possibles (pc)
        pc = []
        for i in range(len(tokens)-3):
            c = (tokens[i],tokens[i+1],tokens[i+2],tokens[i+3])
            pc.append(c)
        ## contextes ajoutés
        for expr in expressions:
            for c in pc:
                if (expr) == (c[1],c[2]):
                    ## ajout à ec_dict
                    key = (c[0],c[3])
                    if key in ec_dict[(c[1],c[2])].keys():
                        ec_dict[(c[1],c[2])][key] += 1
                    else:
                        ec_dict[(c[1],c[2])][key] = 1
                    ## ajout à la liste des contextes
                    if key not in contexts:
                        contexts.append(key)
    ## matrice expression-contexte (EC)
    ec_nrow = len(expressions)
    ec_ncol = len(contexts)
    EC = np.zeros((ec_nrow,ec_ncol), dtype=int)
    for expr in ec_dict:
        i = expressions.index(expr)
        for c in ec_dict[expr]:
            j = contexts.index(c)
            EC[i,j] = ec_dict[expr][c]
    ec_row_names = [re.sub(r'[\']', '', str(e)) for e in expressions]
    ec_col_names = [re.sub(r'[\',]', '', str((str(c[0]),'(.)',str(c[1])))) for c in contexts]
    return pd.DataFrame(EC, index=ec_row_names, columns=ec_col_names)

def _is_mc(mat, t=MC_THRESHOLD):
    for i in range(mat.shape[0]):
        for k in range(mat.shape[1]):
            for j in range(mat.shape[0]):
                for l in range(mat.shape[1]):
                    if np.abs(mat[i,k]/mat[j,k]-mat[i,l]/mat[j,l]) > t:
                        return False
    return True

def _log_posterior_gain(BC, EC):
    BC[BC == 0] = 1
    EC[EC == 0] = 1 # pas le choix apparamment
    r_x = np.sum(BC, 1)
    c_y = np.sum(BC, 0)
    s = np.sum(BC)
    r_p = np.sum(EC, 1)
    c_q = np.sum(EC, 0)
    s_1 = np.sum(EC)
    alpha = ALPHA
    ## somme
    res = (np.sum(r_x*np.log(r_x)) + np.sum(c_y*np.log(c_y)) -
           s*np.log(s) - np.sum(BC*np.log(BC)))
    res += (np.sum(r_p*np.log(r_p)) + np.sum(c_q*np.log(c_q)) -
            s_1*np.log(s_1) - np.sum(EC*np.log(EC)))
    res += alpha*(4*np.sum(BC) - 2*BC.shape[0] - 2*BC.shape[1] - 8)
    return res

def _get_bicluster_pairs(BC):
    pairs = []
    for i in BC.index:
        for j in BC.columns:
            pairs.append((i,j))
    return pairs

def _reduce_corpus(C, BC, N, maximally=False):
    bc_pairs = _get_bicluster_pairs(BC)
    res = C
    if maximally:
        occ = -1
        while occ != 0:
            for pair in bc_pairs:
                res = re.sub(r"\b"+pair[0]+" "+pair[1]+r"\b", N.symbol(), res)
            occ = sum([_count_occ(pair[0]+" "+pair[1], res) for pair in bc_pairs])
    else:
        for pair in bc_pairs:
            res = re.sub(r"\b"+pair[0]+" "+pair[1]+r"\b", N.symbol(), res)
    return res
    
def _count_occ(word, corpus):
    return len(re.findall(r"\b"+word+r"\b", corpus))
    
def _get_best_bicluster(T, C):
    global ignore_mc_ec
    best_bc = None
    max_lpg = -1e9
    ## biclustering sur T
    n = T.shape[0]
    p = T.shape[1]
    clustProg = CoclustInfo(n_row_clusters=n, n_col_clusters=p, n_init=10)
    clustProg.fit(T.as_matrix())
    ## recherche du meilleur bicluster
    for i in range(n):
        for j in range(n):
            bc_row_indices = clustProg.get_row_indices(i)
            bc_col_indices = clustProg.get_col_indices(j)
            bc = T.as_matrix()[np.ix_(bc_row_indices, bc_col_indices)]
            ## BC nul ?
            if np.sum(bc) == 0:
                continue
            bc_row_names = [T.index[x] for x in bc_row_indices]
            bc_col_names = [T.columns[x] for x in bc_col_indices]
            BC = pd.DataFrame(bc, index=bc_row_names, columns=bc_col_names)
            EC = _create_ec(BC, C, T)
            ec = EC.as_matrix()
            ## ignorer EC pour la cohérenche multiplicative
            if ignore_mc_ec:
                ec = np.ones((EC.shape[0],EC.shape[1]))  
            ## BC et EC valid (MC) ?
            if not (_is_mc(bc) and _is_mc(ec)):
                continue
            ## calcul du LPG
            lpg = _log_posterior_gain(bc, EC.as_matrix())
            if lpg > max_lpg:
                max_lpg = lpg
                best_bc = BC.copy()
    return best_bc
    
def _apply_grammar(G, C):
    res = C # derived corpus
    prod_dict = {}
    ## initialisation du dict
    for prod in G.productions():
        prod_dict[prod.lhs().symbol()] = []
    ## remplissage du dict
    for prod in G.productions():
        prod_dict[prod.lhs().symbol()].append(prod)
    ## dérivation de C
    while res.count("_AND_") > 0 or res.count("_OR_") > 0:
        for lhs in prod_dict:
            if "START" in lhs:
                continue
            elif "AND" in lhs and _count_occ(lhs, res) > 0:
                rhs_symbols = [rhs.symbol() for rhs in prod_dict[lhs][0].rhs()]
                res = re.sub(r"\b"+lhs+r"\b", " ".join(rhs_symbols), res)
            elif "OR" in lhs and _count_occ(lhs, res) > 0:
                rhs_symbols = [str(prod.rhs()[0]) for prod in prod_dict[lhs]]
                rhs_probs = [prod.prob() for prod in prod_dict[lhs]]
                rhs_probs /= np.sum(rhs_probs) # normalisation
                ## remplacement
                symbol_count = _count_occ(lhs, res)
                for i in range(symbol_count):
                    rhs = np.random.choice(rhs_symbols, 1, p=rhs_probs)[0]
                    res = res.replace(lhs, rhs, 1)
    return res
    
def _represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False    

def _tuple_to_ec_index(tupl, is_row_index):
    ## True ou False, ex: ('a', 'triangle') ou "('below', -1)
    if is_row_index:
        res = re.sub(r'[\']', '', str(tupl))
    else:
        res = re.sub(r'[\',]', '', str((str(tupl[0]),'(.)',str(tupl[1]))))
    return res
        
def _ec_index_to_tuple(index, is_row_index):
    ## True ou False, ex: "(a, triangle)" ou "below (.) -1"
    if is_row_index:
        res = re.sub(r'[(),]', '', index).split()
    else:
        res = re.sub(r'[().]', '', index).split()
        res[0] = int(res[0]) if _represents_int(res[0]) else res[0]
        res[1] = int(res[1]) if _represents_int(res[1]) else res[1]
    return tuple(res)
    
def _finished(T):
    return T.empty #and max(T.shape) == 1

def _format_nt(s):
    return Nonterminal(s) if "_AND_" in s or "_OR_" in s else s

#
#   ALGORITHME
#------------------------------------

"""Algorithm 4"""
def _postprocessing(G, C):
    print("\npostprocessing...")
    ## suppression de la règle _START_ -> ...
    rules = []
    for prod in G.productions():
        if G.start().symbol() not in prod.lhs().symbol():
            rules.append(prod)
    if len(rules) == 0:
        return G
    ## create an OR symbol S
    S = Nonterminal("_START_")
    sss = {} # single symbol sentences
    ## for each sentence s in C do
    ##   if s is fully reduced to a single symbol x then
    ##   add S -> x to G, or if the rule already exists, increase its weight by 1
    for sentence in sent_tokenize(C):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        t = word_tokenize(sentence)
        if len(t) == 1:
            sss[t[0]] = 1 if not t[0] in sss else sss[t[0]] + 1
    weight_sum = sum([sss[k] for k in sss])
    rules += [ProbabilisticProduction(S, [_format_nt(k)], prob=sss[k]/weight_sum) for k in sss]
    return PCFG(S, rules)

"""Algorithm 3"""
def _attaching(N, G, C, T):
    print("attaching...")
    C_derived = _apply_grammar(G, C)
    ORs = [] # liste des OR (NonTerminal)
    for prod in G.productions():
        nt = prod.lhs()
        if "OR" in nt.symbol() and nt not in ORs:
            ORs.append(nt)
    ## for each OR symbol O in G do
    for O in ORs:
        ## if O leads to a valid expanded bicluster
        ## as well as a posterior gain (Eq.3) larger than a threshold then
        
        #
        #   AND-OR group
        
        group = None
        pos = None # gauche ou droite (impair-False ou pair-True)
        ## récupération du groupe AND-OR de O
        for g in biclusters:
            if O.symbol() in g[1] or O.symbol() in g[2]:
                group = g
                break
        ## récupération de la position de O dand le groupe
        num = int(O.symbol()[4:]) # numéro du OR, ex: "_OR_2" -> 2
        pos = True if num % 2 == 0 else False
        
        #
        #   BC_tilde et BC_tilde_prime
        
        ## création de BC_t (BC_tilde)
        BC_t = biclusters[group].copy()
        ## remplissage de BC_t
        for pair in _get_bicluster_pairs(BC_t):
            BC_t.at[pair] = _count_occ(" ".join(pair), C_derived)
        ## création de BC_t_1 (BC_tilde_prime) (proposed new rule OR -> AND)
        BC_t_1 = BC_t.copy()
        ## . remplissage de BC_t_1
        if pos == False:
            ## new row (OR à gauche)
            new_row = [_count_occ(" ".join((N.symbol(),x)), C) for x in BC_t.columns]
            BC_t_1.loc[N.symbol(),:] = new_row
            BC_t_1 = BC_t_1.astype(int)
        else:
            ## new column (OR à droite)
            new_col = [_count_occ(" ".join((x,N.symbol())), C) for x in BC_t.index]
            BC_t_1.loc[:,N.symbol()] = new_col
            BC_t_1 = BC_t_1.astype(int)
        
        #
        #   EC_tilde et EC_tilde_prime

        ## création et remplissage de EC_t
        EC_t = _create_ec(BC_t, C_derived, _create_t(C_derived))
        ## création de EC_t_1
        EC_t_1 = EC_t.copy()
        ## . ajout des nouvelles lignes de EC_t_1
        if pos == False:
            ## OR à gauche
            new_row_indices = [(N.symbol(),col) for col in BC_t_1.columns]
        else:
            ## OR à droite
            new_row_indices = [(row,N.symbol()) for row in BC_t_1.index]
        ## . remplissage des nouvelles lignes de EC_t_1
        for i in new_row_indices:
            i_str = _tuple_to_ec_index(i, True)
            EC_t_1.loc[i_str,:] = [-1]*EC_t_1.shape[1]
            for j in EC_t_1.columns:
                e, c = " ".join(i), list(_ec_index_to_tuple(j, False)) # expression, contexte
                c = tuple(["" if _represents_int(x) else x for x in c])
                EC_t_1.loc[i_str,j] = _count_occ(" ".join([c[0],e,c[1]]).strip(), C)
        EC_t_1 = EC_t_1.astype(int)
        bc_t_1 = BC_t_1.as_matrix()
        ec_t_1 = EC_t_1.as_matrix()
        bc_t = BC_t.as_matrix()
        ec_t = EC_t.as_matrix()
        
        #
        #   LOG POSTERIOR GAIN DIFFERENCE (Eq.3)
        
        ## BC et EC valid (MC) ?
        if not _is_mc(bc_t_1) and _is_mc(ec_t_1) and _is_mc(bc_t) and _is_mc(ec_t):
            continue
        
        lpg_diff = _log_posterior_gain(bc_t_1, ec_t_1)
        lpg_diff -= _log_posterior_gain(bc_t, ec_t)
        
        if lpg_diff > LPG_DIFF_THRESHOLD:
            print("new rule: %s -> %s" % (O.symbol(),N.symbol()))
            bc = BC_t_1.as_matrix()
            s = np.sum(bc)
            row_prob = np.sum(bc, 1)/s
            col_prob = np.sum(bc, 0)/s
            ## règles
            rules = []
            for prod in G.productions():
                if O.symbol() not in prod.lhs().symbol():
                    rules.append(prod)
            ## ajout des nouvelles règles
            if pos == False:
                ## OR à gauche
                probs = row_prob
                rhs_symbols = [x for x in BC_t.index]+[N]
                for i in range(BC_t_1.shape[0]):
                    rules.append(ProbabilisticProduction(O, [rhs_symbols[i]], prob=probs[i]))
            else:
                ## OR à droite
                probs = col_prob
                rhs_symbols = [x for x in BC_t.columns]+[N]
                for j in range(BC_t_1.shape[1]):
                    rules.append(ProbabilisticProduction(O, [rhs_symbols[j]], prob=probs[j]))
                
            ## mises à jour
            biclusters[group] = BC_t_1.copy() # mise à jour du groupe AND-OR
            G = PCFG(G.start(), rules) # mise à jour de G
            C = _reduce_corpus(C, biclusters[group], N, True) # réduction de C
            T = _create_t(C) # mise à jour de T
            
    return G, C, T

"""Algorithm 2"""
def _learning_by_biclustering(G, C, T):
    print("learning...")
    global biclusters
    global ignore_mc_ec
    
    ## find the valid bicluster Bc in T that leads to the maximal posterior gain (Eq.2)
    BC = None
    
    ## 1er essai
    attempts = 3
    while BC is None and attempts > 0:
        attempts -= 1
        BC = _get_best_bicluster(T, C)
    
    if BC is None:
        ignore_mc_ec = True
    
        ## 2e essai
        attempts = 2
        while BC is None and attempts > 0:
            attempts -= 1
            BC = _get_best_bicluster(T, C)
    
        if BC is None:
            return False, G, C, T, None
        ignore_mc_ec = False
        
    ## create an AND symbol N and two OR symbols A, B
    N = Nonterminal("_AND_"+str(_get_and_symb_index()))
    A = Nonterminal("_OR_"+str(_get_or_symb_index()))
    B = Nonterminal("_OR_"+str(_get_or_symb_index()))
    bc = BC.as_matrix()
    s = np.sum(bc)
    row_prob = np.sum(bc, 1)/s
    col_prob = np.sum(bc, 0)/s
    ## création des règles
    rules = []
    rules += [ProbabilisticProduction(A, [_format_nt(BC.index[i])], prob=row_prob[i])
              for i in range(BC.shape[0])]
    rules += [ProbabilisticProduction(B, [_format_nt(BC.columns[j])], prob=col_prob[j])
              for j in range(BC.shape[1])]
    rules += [ProbabilisticProduction(N, [A, B], prob=1.)]
    ## mises à jour
    G_updated = PCFG(G.start(), G.productions() + rules) # ajout des règles dans G
    C_reduced = _reduce_corpus(C, BC, N) # réduction du corpus
    T_updated = _create_t(C_reduced) # mise à jour de T
    biclusters[(N.symbol(),A.symbol(),B.symbol())] = BC # sauvegarde de BC pour le groupe appris
    return True, G_updated, C_reduced, T_updated, N

"""Algorithm 1"""
def pcfg_bcl(C, alpha=ALPHA, gd_thr=LPG_DIFF_THRESHOLD, mc_thr=MC_THRESHOLD):
    print("\ninitializing...")
    global ALPHA
    global LPG_DIFF_THRESHOLD
    global MC_THRESHOLD
    global and_symb_count
    global or_symb_count
    global ignore_mc_ec
    ALPHA = alpha
    LPG_DIFF_THRESHOLD = gd_thr
    MC_THRESHOLD = mc_thr
    and_symb_count = 0
    or_symb_count = 0
    ignore_mc_ec = False
    
    ## create an empty grammar G
    S = Nonterminal("_START_")
    R = [ProbabilisticProduction(S, [""], prob=1.)]
    G = PCFG(S, R)
    
    T = _create_t(C) # create a table T
    
    ## repeat until no further rule to be learned
    i = 0
    while not _finished(T):
        i += 1
        print("\niter. n° %d" % (i,))
        found, G, C, T, N = _learning_by_biclustering(G, C, T)
        if not found:
            print("NO MORE RULES CAN BE LEARNED")
            break
        G, C, T = _attaching(N, G, C, T)
    G = _postprocessing(G, C)
    print("\n", G) # DEBUG
    return G
