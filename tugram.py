
## honatu.py

import sys
from grammars import read_from_file
from pcfg_bcl import pcfg_bcl

def main(argv):
	corpus = ""
	output_file = "sortie.txt"
	## cas: aucun corpus en entrée
	if len(sys.argv) == 1:
		print("\nAucun fichier passé paramètre.\n")
		print("usage: program.py fichier_entree [fichier_sortie]")
		return
    ## cas: corpus mais pas de fichier de sortie
	elif len(sys.argv) == 2:
		corpus = read_from_file(argv[1]) # corpus
    ## cas: corpus et fichier de sortie
	else:
		corpus = read_from_file(argv[1]) # corpus
		output_file = argv[2]

	gram = pcfg_bcl(corpus) # induction

	## grammaire -> fichier texte
	with open(output_file, "w") as out:
		out.write(str(gram))

if __name__ == "__main__":
    main(sys.argv)