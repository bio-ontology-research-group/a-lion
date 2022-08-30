from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from KG import KG
from jpype import *
import jpype.imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from owlready2 import onto_path, get_ontology
from fuzzywuzzy import fuzz
import json
from joblib import Parallel, delayed 
import multiprocessing
from tqdm import tqdm

#jars_dir = "../src/gateway/build/distributions/gateway/lib/"
#jars = str(str.join(":", [jars_dir+name for name in os.listdir(jars_dir)]))
#startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)

import mowl
mowl.init_jvm("10g")

from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import OWLOntology, OWLLiteral,IRI
from org.semanticweb.owlapi.search import EntitySearcher
from org.semanticweb.owlapi.reasoner import *
from org.semanticweb.elk.reasoner.config import *
from org.semanticweb.elk.owlapi import ElkReasonerFactory, ElkReasonerConfiguration
from org.semanticweb.owlapi.util import *
from org.semanticweb.owlapi.model.parameters import *
from org.semanticweb.elk.reasoner.config import *
from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.reasoner import *
from org.semanticweb.owlapi.reasoner.structural import StructuralReasoner
from org.semanticweb.owlapi.vocab import OWLRDFVocabulary
from org.semanticweb.owlapi.model import *
from org.semanticweb.owlapi.io import *
#from org.semanticweb.owlapi.owllink import *
from org.semanticweb.owlapi.util import *
from org.semanticweb.owlapi.search import *
from org.semanticweb.owlapi.manchestersyntax.renderer import *
from org.semanticweb.owlapi.reasoner.structural import *
from org.apache.jena.rdf.model import *
from org.apache.jena.util import *
from java.util import LinkedHashSet
from java.io import File, PrintWriter, BufferedWriter, FileWriter, FileOutputStream


def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])



#takes an ontology, returns a KG
def Onto2KG(ontology_file_path,temp_name):

	#fout =  PrintWriter( BufferedWriter( FileWriter(temp_name+"_edgelist")))
	#mpout =  PrintWriter( BufferedWriter( FileWriter(temp_name+"_predicates")))
	#meout =  PrintWriter( BufferedWriter( FileWriter(temp_name+"_entitiies")))


	manager = OWLManager.createOWLOntologyManager()
	fac = manager.getOWLDataFactory()


	progressMonitor = ConsoleProgressMonitor()
	config = SimpleConfiguration(progressMonitor)
	f = ElkReasonerFactory()


	#load ontology
	ont = manager.loadOntologyFromOntologyDocument(File(ontology_file_path))
	reasoner = f.createReasoner(ont,config)

	oset =  LinkedHashSet()
	for ax in InferredClassAssertionAxiomGenerator().createAxioms(fac, reasoner):
		manager.addAxiom(ont, ax)

	infered_output = FileOutputStream(ontology_file_path+"_infered")
	manager.saveOntology(ont, infered_output )

	model = ModelFactory.createDefaultModel()
	infile = FileManager.get().open( ontology_file_path+"_infered" )
	model.read(infile,"RDF/XML")

	counter_p = 0
	counter_e = 0
	map_p = {} 
	map_e = {}
	edgelist=[] 
	for stmt in model.listStatements():
		pred = stmt.getPredicate().toString()
		subj = stmt.getSubject()
		obj = stmt.getObject()

		if(str(pred) == "http://www.w3.org/2000/01/rdf-schema#subClassOf"):

			if (subj.isURIResource() and obj.isURIResource()):

				if (pred not in map_p):
					map_p[pred] = counter_p
					counter_p += 1
				if (subj.toString() not in map_e):
					map_e[subj.toString()] = counter_e
					counter_e += 1

				if (obj.toString() not in map_e):
					map_e[obj.toString()] = counter_e
					counter_e += 1

			
				predid = str(map_p[pred])
				subjid = str(map_e[subj.toString()])
				objid = str(map_e[obj.toString()])
			
				edgelist.append([pred,subj.toString(),obj.toString()])
				#edgelist.append([subjid,objid,predid])
				#fout.println(subjid+"\t"+objid+"\t"+predid)

	#for k in map_e:
	#  meout.println(k+"\t"+str(map_e[k]))
	
	#for k in map_p:
	#  mpout.println(k+"\t"+str(map_p[k]))


	#fout.flush()
	#fout.close()
	#meout.flush()
	#meout.close()
	#mpout.flush()
	#mpout.close()

	newKG = KG()
	newKG.load_triples_from_dic(edgelist)
	return newKG



	print("done")
#test passed
#kg = Onto2KG("/home/alghsm0a/AgreementMakerDeep-main/conf/oaei-resources/source.owl", "test")
#print(kg)


def lex_ma_from_dic(lab,k,dic,r):
	ls = []
	for l in dic:
		ratio = fuzz.ratio(lab,l)
		if(ratio>r):
			ls.append([k,dic[l]])
			#print(lab,l,ratio)
	return(ls)


#takes 2 ontologies, return the lexical matching pairs of alignments
def LexicalMatch(source, target, txt):

	print("load ontology 1")
	onto1 = get_ontology(source)
	onto1.load()

	base1 = onto1.base_iri
	print("parse labels for ontology 1")
	ont1_label2class = {}
	for cl in onto1.classes():        
		labels = cl.label
		ont1_label2class[cl.name.lower()]= cl.iri
		for lab in labels:
			ont1_label2class[lab.lower()]= cl.iri

	with open(txt+'_source.json','w') as f:
		json.dump(ont1_label2class,f)

	print("load ontology 2")
	onto2 = get_ontology(target)
	onto2.load()

	base2 = onto2.base_iri
	print("parse labels for ontology 2")
	ont2_label2class = {}
	for cl in onto2.classes():        
		labels = cl.label
		ont2_label2class[cl.name.lower()]= cl.iri
		for lab in labels:
			ont2_label2class[lab.lower()]= cl.iri


	with open(txt+'_target.json','w') as f:
		json.dump(ont2_label2class,f)

	print("start lexical alignments")
	alignments = []

	


	print("start Parallelizing lexical matxhing")

	accepted_ratio = 96

	while(len(alignments)<10):

		keys = ont1_label2class.keys()
		num_core = multiprocessing.cpu_count()
		Result = Parallel(n_jobs=1)(delayed(lex_ma_from_dic)(k,ont1_label2class[k], ont2_label2class,accepted_ratio) for k in tqdm(keys) )
		for  i in range(len(keys)):
			alignments+=Result[i]
		print(len(alignments), "alignment found with accepted ratio", accepted_ratio)
		accepted_ratio-=10



	with open(txt+'.json','w') as f:
		json.dump(alignments,f)



	return alignments 


# test
#A = LexicalMatch("/home/alghsm0a/A-LIOn/anatomy-dataset/human.owl", "/home/alghsm0a/A-LIOn/anatomy-dataset/mouse.owl","anatomy")
