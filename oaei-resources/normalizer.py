from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model.parameters import Imports
from org.semanticweb.owlapi.model import OWLClass, OWLClassExpression, IRI
from uk.ac.manchester.cs.owl.owlapi import OWLObjectSomeValuesFromImpl

from mowl.reasoning.normalize import GCI0, GCI1, GCI2, GCI3, GCI0_BOT, GCI1_BOT, GCI3_BOT

from jpype import *

class Normalizer():

    def __init__(self):
        self.virtual_entity_prefix = "http://mowl/virtual_"
        self.entity_counter = 0
        return

    
    
    def normalize(self, ontology):
        
        self.ont_manager = OWLManager.createOWLOntologyManager()
        self.data_factory = self.ont_manager.getOWLDataFactory()

        self.bottom_class = self.data_factory.getOWLClass(IRI.create("http://www.w3.org/2002/07/owl#Nothing"))
        self.top_class = self.data_factory.getOWLClass(IRI.create("http://www.w3.org/2002/07/owl#Thing"))

        imports = Imports.fromBoolean(False)
        axioms = ontology.getAxioms(imports)
    
        print(f"INFO: Number of axioms: {len(axioms)}")

        gcis = {
            "gci0": [],
            "gci1": [],
            "gci2": [],
            "gci3": [],
            "gci0_bot": [],
            "gci1_bot": [],
            "gci3_bot": []
        }

        for ax in axioms:
            new_axioms = self.parse_axiom(ax)
            for ax_name, norm_ax in new_axioms:
                gcis[ax_name].append(norm_ax)
        return gcis
  


    def parse_axiom(self, axiom):
        axiom_type = axiom.getAxiomType().getName()
        
        if axiom_type == "EquivalentClasses":

            expr_list= list(axiom.getClassExpressionsAsList())
            fst = find_elem(lambda x: x.getClassExpressionType().getName() == "Class", expr_list)

            if fst:
                others = list(filter(lambda x: x!= fst, expr_list))
                axioms = flat_map(lambda x: self.parse_equiv_class_axiom(fst, x), others)
                return axioms
            else:
                print(f"Equivalent classes axiom too complex {axiom}")
                return []
          
        elif axiom_type == "SubClassOf":
            if axiom.getSubClass().getClassExpressionType().getName() == "Class":
                return self.parse_subclass_axiom(axiom.getSubClass(), axiom.getSuperClass())
            elif ax.getSuperClass().getClassExpressionType().getName() == "Class":
                return self.parse_subclass_axiom_complex(ax.getSubClass(), ax.getSuperClass())
            else:
                print(f"Sublcass Axiom too complex {axiom}")
                return []
       
        elif axiom_type ==  "DisjointClasses":
            expr_list = list(axiom.getClassExpressionsAsList())
            fst = find_elem(lambda x: x.getClassExpressionType().getName() == "Class", expr_list)

            if fst:
                others = list(filter(lambda x: x != fst, expr_list))
                axioms = flat_map(lambda x: self.parseDisjointnessAxiom(fst, x), others)
                return axioms
            else:
                println(f"Disjointness axiom too complex {axiom}")
                return []
          
        
        elif axiom_type == "Declaration":
            return []
        elif axiom_type == "AnnotationAssertion":
            return []
        else:
            print(f"Not parsing axiom {axiom_type}")
            return []


    
    def parse_equiv_class_axiom(self, leftExpr: OWLClass, rightSideExpr: OWLClassExpression):
        
        rightSideType = rightSideExpr.getClassExpressionType().getName()

        if rightSideType == "Class": #A equiv B
            rightOWLClass = rightSideExpr
            axiom1 = "gci0", GCI0(self.create_gci0(leftExpr, rightOWLClass))
            axiom2 = "gci0", GCI0(self.create_gci0(rightOWLClass, leftExpr))
            return [axiom1, axiom2]
            
        else:
            axiomsOneDirection = self.parse_subclass_axiom(leftExpr, rightSideExpr, equivalent = True)
            axiomsOtherDirecion = self.parse_subclass_axiom_complex(rightSideExpr, leftExpr, equivalent = True)
            return axiomsOneDirection + axiomsOtherDirecion

    def parseDisjointnessAxiom(self,go_class: OWLClass, rightSideExpr: OWLClassExpression):
        rightSideType = rightSideExpr.getClassExpressionType().getName()

        if rightSideType == "Class":
            return [("gci1_bot", GCI1_BOT(self.create_gci1(go_class, rightSideExpr, self.bottom_class)))]
        else:
            print(f"Disjointness axiom: complex expression {rightSideType}")
            return []

    def parse_subclass_axiom(self, go_class: OWLClass, superclass: OWLClassExpression, equivalent = False):
        rightSideType = superclass.getClassExpressionType().getName()

        if rightSideType == "Class":
            rightOWLClass = superclass
            axiom1 = "gci0", GCI0(self.create_gci0(go_class, rightOWLClass))
            return [axiom1]

        elif rightSideType == "ObjectSomeValuesFrom":
            rightSideEx = superclass
            property_ = rightSideEx.getProperty()
            filler = rightSideEx.getFiller()

            fillerType = filler.getClassExpressionType().getName()

            if fillerType == "Class":
                return [("gci2", GCI2(self.create_gci2(go_class, property_, filler)))]
            else:
                print(f"Subclass axiom: filler type not suported. Type is {fillerType}")
                return []
        elif rightSideType == "ObjectIntersectionOf":
            rightSideOperands = list(superclass.getOperands())
            axioms = flat_map(lambda x: self.parse_subclass_axiom(go_class, x), rightSideOperands)
            return axioms
        else:
            print(f"Subclass axiom: rightside not supported. Type is {rightSideType}")
            return []

    def parse_subclass_axiom_complex(self, subClass: OWLClassExpression, go_class: OWLClass, equivalent = False):
        leftSideType = subClass.getClassExpressionType().getName()

        if leftSideType == "Class":
            leftOWLClass = subClass
            axiom1 = "gci0", GCI0(crate_gci0(leftOWLClass, go_class))
            return [axiom1]

        elif leftSideType == "ObjectIntersectionOf":
            leftSideOperands = list(subClass.getOperands())
            leftSideTypes = list(map(lambda x: x.getClassExpressionType().getName(), leftSideOperands))

            num_operands = len(leftSideOperands)
            if num_operands == 1:
                return self.parse_subclass_axiom_complex(leftSideOperands[0], go_class, equivalent = equivalent)
            elif num_operands == 2:
                if all(map(lambda x: x == "Class", leftSideTypes)):
                    axiom = "gci1", GCI1(self.create_gci1(leftSideOperands[0], leftSideOperands[1], go_class))
                    return [axiom]
                else:
                    print(f"Subclass complex axiom: left side expression not supported {leftSideTypes}")
                    return []
            else:
                print(f"Intersection too large: {num_operands}")
                return []

        elif leftSideType ==  "ObjectSomeValuesFrom":
            leftSideEx = subClass
            property_ = leftSideEx.getProperty()
            filler = leftSideEx.getFiller()

            fillerType = filler.getClassExpressionType().getName()

            if fillerType == "Class":
                axiom = "gci3", GCI3(self.create_gci3(property_, filler, go_class))
                return [axiom]
            else:
                print(f"Subclass complex axiom: existential filler in left side not suported. Type is {fillerType}")
                return []
        else:
            print(f"Subclass complex axiom: left side not supported. Type is {leftSideType}")
            return []

    def create_gci0(self, subclass, superclass):
        axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, superclass)
        return axiom

    def create_gci1(self, left_sub, right_sub, superclass):
        right_side = self.data_factory.getOWLObjectIntersectionOf(left_sub, right_sub, superclass)
        axiom = self.data_factory.getOWLSubClassOfAxiom(right_side, superclass)
        return axiom

    def create_gci2(self, subclass, obj_property, filler):
        left_side = self.data_factory.getOWLObjectSomeValuesFrom(obj_property, filler)
        axiom = self.data_factory.getOWLSubClassOfAxiom(subclass, left_side)
        return axiom
    
    def create_gci3(self, obj_property, filler, superclass):
        right_side = self.data_factory.getOWLObjectSomeValuesFrom(obj_property, filler)
        axiom = self.data_factory.getOWLSubClassOfAxiom(right_side, superclass)
        return axiom
        




def find_elem(condition, iterable):
    for elem in iterable:
        if condition(elem):
            return elem
    return None


def flat_map(function, iterable):
    mapped_list = list(map(function, iterable))

    flattened_list = []
    for elem in mapped_list:
        flattened_list += elem
    return flattened_list
