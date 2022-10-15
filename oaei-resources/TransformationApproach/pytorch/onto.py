import sys
import os
import json
from fuzzywuzzy import fuzz
import multiprocessing
#from jpype import *
#import jpype.imports
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from owlready2 import onto_path, get_ontology

#
#from joblib import Parallel, delayed 

#from tqdm import tqdm
#import numpy as np

import mowl
mowl.init_jvm("10g")
from mowl.owlapi import OWLAPIAdapter

from org.semanticweb.owlapi.search import EntitySearcher
from org.semanticweb.owlapi.reasoner import ConsoleProgressMonitor, SimpleConfiguration
from org.semanticweb.elk.owlapi import ElkReasonerFactory, ElkReasonerConfiguration
from java.io import File
from java.util import ArrayList, LinkedHashSet
from org.semanticweb.owlapi.model import IRI
# from org.semanticweb.owlapi.apibinding import OWLManager
# from org.semanticweb.owlapi.model import OWLOntology, OWLLiteral,IRI, AxiomType
# 
#
# from org.semanticweb.elk.reasoner.config import *
# 
# from org.semanticweb.owlapi.util import *
# from org.semanticweb.owlapi.model.parameters import *
# from org.semanticweb.elk.reasoner.config import *
# from org.semanticweb.owlapi.apibinding import OWLManager
# from org.semanticweb.owlapi.reasoner import *
# from org.semanticweb.owlapi.reasoner.structural import StructuralReasoner
# from org.semanticweb.owlapi.vocab import OWLRDFVocabulary
# from org.semanticweb.owlapi.model import *
# from org.semanticweb.owlapi.io import *
# #from org.semanticweb.owlapi.owllink import *
# from org.semanticweb.owlapi.util import *
# from org.semanticweb.owlapi.search import *
# from org.semanticweb.owlapi.manchestersyntax.renderer import *
# from org.semanticweb.owlapi.reasoner.structural import *
# from org.apache.jena.rdf.model import *
# from org.apache.jena.util import *
# from java.util import LinkedHashSet, List, ArrayList, Arrays
# from java.io import PrintWriter, BufferedWriter, FileWriter, FileOutputStream
# from com.clarkparsia.owlapi.explanation import BlackBoxExplanation
# from com.clarkparsia.owlapi.explanation import HSTExplanationGenerator


def lex_ma_from_dic_pool(dic1, dic2, r, label):
    k = dic1[label]
    return lex_ma_from_dic(label, k, dic2, r)

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
def lexical_match(source, target, out_root):

    ont1_label2class = {}
    ont2_label2class = {}
    print("load ontology 1")

    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    fac = adapter.data_factory

    
    ont1 = manager.loadOntologyFromOntologyDocument(File(source))
    ont1_classes = ont1.getClassesInSignature(True)
    for cl in ont1_classes:
        class_iri = str(cl.toString()).replace("<","").replace(">","")
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
    with open(out_root+'_source_label2class.json','w') as f:
            json.dump(ont1_label2class,f)

    ont2 = manager.loadOntologyFromOntologyDocument(File(target))
    ont2_classes = ont2.getClassesInSignature(True)
    for cl in ont2_classes:
        class_iri = str(cl.toString()).replace("<","").replace(">","")
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
    with open(out_root+'_target_label2class.json','w') as f:
            json.dump(ont2_label2class,f)

    print("start lexical alignments")
    alignments = []
    print("start Parallelizing lexical matching")

    accepted_ratio = 96
    ont1_cls = [c for c in ont1_classes]
    ont2_cls = [c for c in ont2_classes]
    min_entities = min(len(ont1_cls), len(ont2_cls))//2

    min_alignments = None
    #min_entities = 100
    while(len(alignments) <= min_entities and accepted_ratio>60):

        keys = ont1_label2class.keys()
        num_core = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(num_core)
        print("Starting pool...")
        Result = [pool.apply_async(lex_ma_from_dic_pool,
                                   args=(ont1_label2class, ont2_label2class, accepted_ratio, k)) for k in keys]
        Result = [p.get() for p in Result]
        # Result = Parallel(n_jobs=1)(delayed(lex_ma_from_dic)(k,ont1_label2class[k], ont2_label2class,accepted_ratio) for k in keys )
        for  i in range(len(keys)):
            alignments+=Result[i]
            
        if min_alignments is None:
            min_alignments = len(alignments)//20 +1
        print(len(alignments), "alignment found with accepted ratio", accepted_ratio)
        accepted_ratio-=10


    if min_alignments is None:
        min_alignments = 0

    print(f"TOTAL ALIGNMENTS =",  len(alignments))
    print(f"TOTAL MIN ALIGNMENTS =",  min_alignments)

    return alignments#, min_alignments





def remove_inconsistent_alignments(source, target, predicted_alignment):
    #load ontologies
    adapter = OWLAPIAdapter()
    manager = adapter.owl_manager
    fac = adapter.data_factory


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
