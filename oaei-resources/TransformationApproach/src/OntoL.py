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
import numpy as np
#jars_dir = "../src/gateway/build/distributions/gateway/lib/"
#jars = str(str.join(":", [jars_dir+name for name in os.listdir(jars_dir)]))
#startJVM(getDefaultJVMPath(), "-ea",  "-Djava.class.path=" + jars,  convertStrings=False)

import mowl
mowl.init_jvm("10g")

from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import OWLOntology, OWLLiteral,IRI, AxiomType
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
from java.util import LinkedHashSet, List, ArrayList, Arrays
from java.io import File, PrintWriter, BufferedWriter, FileWriter, FileOutputStream
from com.clarkparsia.owlapi.explanation import BlackBoxExplanation
from com.clarkparsia.owlapi.explanation import HSTExplanationGenerator




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
def Onto2KG(ontology_file_path):

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
            #print(k, dic[l])
            ls.append([k,dic[l]])
			#print(lab,l,ratio)
    return(ls)


#takes 2 ontologies, return the lexical matching pairs of alignments
def LexicalMatch(source, target, txt):

    ont1_label2class = {}
    ont2_label2class = {}
    print("load ontology 1")

    try:
        onto1 = get_ontology(source+"_infered")
        onto1.load()
        
        base1 = onto1.base_iri
        print("parse labels for ontology 1")
        for cl in onto1.classes():		
            labels = cl.label
            ont1_label2class[cl.name.lower()]= cl.iri
            for lab in labels:
                ont1_label2class[lab.lower()]= cl.iri
    except:
        print("owlready fails to parse source ontology, trying owlapi ... ")

    manager = OWLManager.createOWLOntologyManager()
    fac = manager.getOWLDataFactory()
    ont1 = manager.loadOntologyFromOntologyDocument(File(source))
    for cl in ont1.getClassesInSignature(True):
        class_iri = str(cl.toString().replace("<","").replace(">",""))
        id_index_start = max(class_iri.find("/"),class_iri.find("#"))
        class_id = class_iri[id_index_start:]
        ont1_label2class[class_id.lower()]= class_iri
        for lab in EntitySearcher.getAnnotationObjects(cl, ont1, fac.getRDFSLabel()):
            #print(lab.getValue())
            if (lab.getValue().isLiteral()): #isOfType(OWLLiteral)
                labs = lab.getValue().getLiteral()
                #print("label in get litral", labs )
                ont1_label2class[str(labs).lower()]= class_iri
            if ("http://www.w3.org/2004/02/skos/core#prefLabel" in lab.getProperty().toString()):
                labs = lab.getValue().getLiteral()
                #print("label in prefLabel", labs )
                ont1_label2class[str(labs.lower())]= class_iri

    if(len(ont1_label2class.keys())>5):
            print("not enuogh labels in the source ontology")
    with open(txt+'_source.json','w') as f:
            json.dump(ont1_label2class,f)
    try:
        print("load ontology 2")
        onto2 = get_ontology(target+"_infered")
        onto2.load()

        base2 = onto2.base_iri
        print("parse labels for ontology 2")
        for cl in onto2.classes():		
            labels = cl.label
            ont2_label2class[cl.name.lower()]= cl.iri
            for lab in labels:
                ont2_label2class[lab.lower()]= cl.iri
    except:
        print("owlready fails to parse target ontology, trying owlapi ... ")
    ont2 = manager.loadOntologyFromOntologyDocument(File(target))
    for cl in ont2.getClassesInSignature(True):
        class_iri = str(cl.toString().replace("<","").replace(">",""))
        id_index_start = max(class_iri.find("/"),class_iri.find("#"))
        class_id = class_iri[id_index_start:]
        ont2_label2class[class_id.lower()]= class_iri
        for lab in EntitySearcher.getAnnotationObjects(cl, ont2, fac.getRDFSLabel()):
            #print(lab.getValue())
            if (lab.getValue().isLiteral()): #isOfType(OWLLiteral)
                labs = lab.getValue().getLiteral()
                #print("label in get litral", labs )
                ont2_label2class[str(labs).lower()]= class_iri
            if ("http://www.w3.org/2004/02/skos/core#prefLabel" in lab.getProperty().toString()):
                labs = lab.getValue().getLiteral()
                #print("label in prefLabel", labs )
                ont2_label2class[str(labs).lower()]= class_iri

    if(len(ont2_label2class.keys())>5):
        print("not enuogh labels in the target ontology")
    with open(txt+'_target.json','w') as f:
            json.dump(ont2_label2class,f)

    print("start lexical alignments")
    alignments = []
    print("start Parallelizing lexical matching")

    accepted_ratio = 96
    ont1_cls = [c for c in onto1.classes()]
    ont2_cls = [c for c in onto2.classes()]
    min_entities = min(len(ont1_cls), len(ont2_cls))//2

    min_alignments = None
    #min_entities = 10
    while(len(alignments) <= min_entities and accepted_ratio>60):

        keys = ont1_label2class.keys()
        num_core = multiprocessing.cpu_count()
        Result = Parallel(n_jobs=1)(delayed(lex_ma_from_dic)(k,ont1_label2class[k], ont2_label2class,accepted_ratio) for k in tqdm(keys) )
        for  i in range(len(keys)):
            alignments+=Result[i]
            
        if min_alignments is None:
            min_alignments = len(alignments)//20 +1
        print(len(alignments), "alignment found with accepted ratio", accepted_ratio)
        accepted_ratio-=10



    #with open(txt+'.json','w') as f:
    #    json.dump(alignments,f)

    if min_alignments is None:
        min_alignments = 0

    print(f"TOTAL ALIGNMENTS =",  len(alignments))
    print(f"TOTAL MIN ALIGNMENTS =",  min_alignments)

    return alignments, min_alignments





def removeInconsistincyAlignmnets(source, target, predicted_alignment):
    #load ontologies
    manager = OWLManager.createOWLOntologyManager()
    fac = manager.getOWLDataFactory()


    progressMonitor = ConsoleProgressMonitor()
    config = SimpleConfiguration(progressMonitor)
    f = ElkReasonerFactory()
    onts = ArrayList()
    onts.add(manager.loadOntologyFromOntologyDocument(File(source)))
    onts.add(manager.loadOntologyFromOntologyDocument(File(target)))

    #merge ontologuies
    mperged = IRI.create("http://ontology_url/")
    mergedOnt = manager.createOntology(mperged,LinkedHashSet(onts))
    #add alignment as equivilant classes
    try:
        reasoner = f.createReasoner(mergedOnt,config)
        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
    except:
        print("Error,,, Ontologies are not EL consistent, can't apply INS ")
        return []
    added = []
    not_added = []
    added_equiv = []
    not_added_equiv = []
    #problematic_axioms_score={}
    for align in predicted_alignment:
        iri1 = fac.getOWLClass(IRI.create(align[0]))
        iri2 = fac.getOWLClass(IRI.create(align[1]))
        eq = ArrayList()
        eq.add(iri1)
        eq.add(iri2)
        equiv_axiom = fac.getOWLEquivalentClassesAxiom(LinkedHashSet(eq))
        try:
            manager.addAxiom(mergedOnt, equiv_axiom)
            reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
            added.append(align)
            added_equiv.append(equiv_axiom)
        except:
            print("OPSS")
            score = 0 
            might_be_removed_equiv = []
            might_be_removed = []
            print("problem in reasoning")
            for i in range(len(added_equiv)):
                manager.removeAxiom(mergedOnt, added_equiv[i])
            try:
                reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
                for i in range(len(added)):
                    try:
                        manager.addAxiom(mergedOnt, added_equiv[i])
                        reasoner.precomputeInferences(InferenceType.CLASS_HIERARCHY)
                    except:
                        score +=1
                        #if(added[i] not in problematic_axioms_score):
                        #	problematic_axioms_score[added[i]]=0
                        #problematic_axioms_score[added[i]]+=1
                        manager.removeAxiom(mergedOnt, added_equiv[i])
                        might_be_removed_equiv.append([added_equiv[i]])
                        might_be_removed.append([added[i]])

                if(score>1):
                    not_added.append(align)
                    not_added_equiv.append(equiv)
                    manager.removeAxiom(mergedOnt,equiv)
                    for readd in might_be_removed_equiv:
                        manager.addAxiom(mergedOnt, readd)
                else:
                    manager.removeAxiom(mergedOnt,might_be_removed_equiv[0])
                    added.remove(might_be_removed[0])
                    added_equiv.remove(might_be_removed_equiv[0])
                    not_added.append(might_be_removed[0])
                    not_added_equiv.append(might_be_removed_equiv[0])


            except:
                print("problematic in its own")
                #if(align not in problematic_axioms_score):
                #	problematic_axioms_score[added[i]]=0
                #problematic_axioms_score[align]+=1
                manager.removeAxiom(mergedOnt, equiv_axiom)
                not_added.append(align)
                not_added_equiv.append(equiv_axiom)


    print("added", len(added), "from", len(predicted_alignment))
    print("not added", len(not_added))

    inc_negatives = []
    for a in not_added:
            inc_negatives.append(a)


    unsatisfaiable_classes =  ArrayList()
    for cl in mergedOnt.getClassesInSignature(True):
            if(not reasoner.isSatisfiable(cl)):
                    unsatisfaiable_classes.add(cl)
    print("number of unsatisfaiable classes",unsatisfaiable_classes.size() )


    exp = BlackBoxExplanation(mergedOnt, f, reasoner)
    multExplanator = HSTExplanationGenerator(exp)

    if(unsatisfaiable_classes.size()>0):
        for cl in unsatisfaiable_classes:
            print("##############class to explain", cl.toString())
            explanation = multExplanator.getExplanation(cl)
            #print(explanation.toString())
            for axiom in explanation:
                print("--------------guilty axiom--------------")
                if(axiom.isOfType(AxiomType.EQUIVALENT_CLASSES)):
                    #print(axiom.toString())
                    classes = axiom.getNamedClasses()
                    align = [str(classes[0].toString()),str(classes[1].toString())]
                    if (align in added):
                        inc_negatives.append(align)


    return inc_negatives



# test
#A = LexicalMatch("/home/alghsm0a/A-LIOn/anatomy-dataset/human.owl", "/home/alghsm0a/A-LIOn/anatomy-dataset/mouse.owl","anatomy")
