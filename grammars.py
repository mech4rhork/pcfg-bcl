
## grammars.py

from nltk.grammar import PCFG, Nonterminal, ProbabilisticProduction
from nltk.parse import generate

def baseline(depth=5, n=500):
    ## symboles non terminaux
    S = Nonterminal("S")
    NP = Nonterminal("NP")
    VP = Nonterminal("VP")
    PP = Nonterminal("PP")
    Det = Nonterminal("Det")
    Vt = Nonterminal("Vt")
    Vc = Nonterminal("Vc")
    Vi = Nonterminal("Vi")
    N = Nonterminal("N")
    P = Nonterminal("P")
    ## règles de production probabilistes
    R = [ProbabilisticProduction(S, [NP, VP], prob=1.),
         ProbabilisticProduction(NP, [Det, N], prob=1.),
         ProbabilisticProduction(VP, [Vt, NP], prob=1/3),
         ProbabilisticProduction(VP, [Vc, PP], prob=1/3),
         ProbabilisticProduction(VP, [Vi], prob=1/3),
         ProbabilisticProduction(PP, [P, NP], prob=1.),
         ProbabilisticProduction(Det, ["a"], prob=.5),
         ProbabilisticProduction(Det, ["the"], prob=.5),
         ProbabilisticProduction(Vt, ["touches"], prob=.5),
         ProbabilisticProduction(Vt, ["covers"], prob=.5),
         ProbabilisticProduction(Vi, ["rolls"], prob=.5),
         ProbabilisticProduction(Vi, ["bounces"], prob=.5),
         ProbabilisticProduction(Vc, ["is"], prob=1.),
         ProbabilisticProduction(N, ["circle"], prob=1/3),
         ProbabilisticProduction(N, ["square"], prob=1/3),
         ProbabilisticProduction(N, ["triangle"], prob=1/3),
         ProbabilisticProduction(P, ["above"], prob=.5),
         ProbabilisticProduction(P, ["below"], prob=.5)]
    G = PCFG(S, R) # grammaire
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return G, C
    
def num_agr(depth=5, n=500):
    ## symboles non terminaux
    S = Nonterminal("S")
    NP = Nonterminal("NP")
    NP_sg = Nonterminal("NP_sg")
    NP_pl = Nonterminal("NP_pl")
    VP_sg = Nonterminal("VP_sg")
    VP_pl = Nonterminal("VP_pl")
    PP = Nonterminal("PP")
    Det = Nonterminal("Det")
    Vt_sg = Nonterminal("Vt_sg")
    Vt_pl = Nonterminal("Vt_pl")
    Vc_sg = Nonterminal("Vc_sg")
    Vc_pl = Nonterminal("Vc_pl")
    Vi_sg = Nonterminal("Vi_sg")
    Vi_pl = Nonterminal("Vi_pl")
    N_sg = Nonterminal("N_sg")
    N_pl = Nonterminal("N_pl")
    P = Nonterminal("P")
    ## règles de production probabilistes
    R = [ProbabilisticProduction(S, [NP_sg, VP_sg], prob=.5),
         ProbabilisticProduction(S, [NP_pl, VP_pl], prob=.5),
         ProbabilisticProduction(NP, [NP_sg], prob=.5),
         ProbabilisticProduction(NP, [NP_pl], prob=.5),
         ProbabilisticProduction(NP_sg, [Det, N_sg], prob=1.),
         ProbabilisticProduction(NP_pl, [Det, N_pl], prob=1.),
         ProbabilisticProduction(VP_sg, [Vt_sg, NP], prob=1/3),
         ProbabilisticProduction(VP_sg, [Vc_sg, PP], prob=1/3),
         ProbabilisticProduction(VP_sg, [Vi_sg], prob=1/3),
         ProbabilisticProduction(VP_pl, [Vt_pl, NP], prob=1/3),
         ProbabilisticProduction(VP_pl, [Vc_pl, PP], prob=1/3),
         ProbabilisticProduction(VP_pl, [Vi_pl], prob=1/3),
         ProbabilisticProduction(PP, [P, NP], prob=1.),
         ProbabilisticProduction(Det, ["the"], prob=1.),
         ProbabilisticProduction(Vt_sg, ["touches"], prob=.5),
         ProbabilisticProduction(Vt_sg, ["covers"], prob=.5),
         ProbabilisticProduction(Vt_pl, ["touch"], prob=.5),
         ProbabilisticProduction(Vt_pl, ["cover"], prob=.5),
         ProbabilisticProduction(Vc_sg, ["is"], prob=1.),
         ProbabilisticProduction(Vc_pl, ["are"], prob=1.),
         ProbabilisticProduction(Vi_sg, ["rolls"], prob=.5),
         ProbabilisticProduction(Vi_sg, ["bounces"], prob=.5),
         ProbabilisticProduction(Vi_pl, ["roll"], prob=.5),
         ProbabilisticProduction(Vi_pl, ["bounce"], prob=.5),        
         ProbabilisticProduction(N_sg, ["circle"], prob=1/3),
         ProbabilisticProduction(N_sg, ["square"], prob=1/3),
         ProbabilisticProduction(N_sg, ["triangle"], prob=1/3),
         ProbabilisticProduction(N_pl, ["circles"], prob=1/3),
         ProbabilisticProduction(N_pl, ["squares"], prob=1/3),
         ProbabilisticProduction(N_pl, ["triangles"], prob=1/3),
         ProbabilisticProduction(P, ["above"], prob=.5),
         ProbabilisticProduction(P, ["below"], prob=.5)]
    G = PCFG(S, R) # grammaire
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return G, C
    
def english_0(depth=5, n=500):
    ## symboles non terminaux
    S = Nonterminal("S")
    NP = Nonterminal("NP")
    VP = Nonterminal("VP")
    PP = Nonterminal("PP")
    Det = Nonterminal("Det")
    Vt = Nonterminal("Vt")
    Vi = Nonterminal("Vi")
    N = Nonterminal("N")
    P = Nonterminal("P")
    ## règles de production probabilistes
    R = [ProbabilisticProduction(S, [NP, VP], prob=1.),
         ProbabilisticProduction(NP, [Det, N], prob=1.),
         ProbabilisticProduction(VP, [Vt, NP], prob=.5),
         ProbabilisticProduction(VP, [Vi, PP], prob=.5),
         ProbabilisticProduction(PP, [P, NP], prob=1.),
         ProbabilisticProduction(Det, ["a"], prob=1.),
         ProbabilisticProduction(Vt, ["touches"], prob=1.),
         ProbabilisticProduction(Vi, ["is"], prob=1.),
         ProbabilisticProduction(N, ["circle"], prob=1/3),
         ProbabilisticProduction(N, ["square"], prob=1/3),
         ProbabilisticProduction(N, ["triangle"], prob=1/3),
         ProbabilisticProduction(P, ["above"], prob=.5),
         ProbabilisticProduction(P, ["below"], prob=.5)]
    G = PCFG(S, R) # grammaire
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return G, C

def langley_1(depth=5, n=500):
    ## symboles non terminaux
    S = Nonterminal("S")
    NP = Nonterminal("NP")
    VP = Nonterminal("VP")
    AP = Nonterminal("AP")
    Adj = Nonterminal("Adj")
    Det = Nonterminal("Det")
    Vt = Nonterminal("Vt")
    Vi = Nonterminal("Vi")
    N = Nonterminal("N")
    ## règles de production probabilistes
    R = [ProbabilisticProduction(S, [NP, VP], prob=1.),
         ProbabilisticProduction(VP, [Vi], prob=.5),
         ProbabilisticProduction(VP, [Vt, NP], prob=.5),
         ProbabilisticProduction(NP, [Det, N], prob=.5),
         ProbabilisticProduction(NP, [Det, AP, N], prob=.5),
         ProbabilisticProduction(AP, [Adj], prob=.5),
         ProbabilisticProduction(AP, [Adj, AP], prob=.5),
         ProbabilisticProduction(Det, ["the"], prob=1.),
         ProbabilisticProduction(Vt, ["saw"], prob=.5),
         ProbabilisticProduction(Vt, ["heard"], prob=.5),
         ProbabilisticProduction(Vi, ["ate"], prob=.5),
         ProbabilisticProduction(Vi, ["slept"], prob=.5),
         ProbabilisticProduction(N, ["cat"], prob=.5),
         ProbabilisticProduction(N, ["dog"], prob=.5),
         ProbabilisticProduction(Adj, ["big"], prob=.5),
         ProbabilisticProduction(Adj, ["old"], prob=.5)]
    G = PCFG(S, R) # grammaire
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return G, C

def langley_2(depth=5, n=500):
    G = PCFG.fromstring("""
    S -> NP VP [1.0]
    VP -> V NP [1.0]
    NP -> Det N [0.5] | Det N RC [0.5]
    RC -> Rel VP [1.0]
    Det -> 'the' [0.5] | 'a' [0.5]
    V -> 'saw' [0.5] | 'heard' [0.5]
    N -> 'cat' [0.3333] | 'dog' [0.3333] | 'mouse' [0.3333]
    Rel -> 'that' [1.0]
    """)
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return G, C

def generate_from_grammar(G, depth=50, n=999):
    C = "" # corpus
    ## toutes les phrases possibles
    print("\n")
    for n, sent in enumerate(generate.generate(G, depth=depth, n=n), 1):
        s = ' '.join(sent)
        C += s + '. '
        print('%3d. %s%s' % (n, s, '.'))
    return C
    
def read_from_file(filename):
    with open(filename, "r") as file:
        C = file.read().replace("\n", " ")
    return C