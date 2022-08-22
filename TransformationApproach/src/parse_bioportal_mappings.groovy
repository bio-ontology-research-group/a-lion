

// this code parse the ontology mappings provided by BioPortal
// pass your api key as an argument (get an api key from http://bioportal.bioontology.org/accounts/new) 

import groovy.json.JsonSlurper

def apiKey = args[0]
def url = 'http://data.bioontology.org/mappings?ontologies=FMA%2CMA&apikey='+apiKey

def nextpage = '&page=1'
def jsonSlurper = new JsonSlurper()
def mappingDict
def command
def proc

while (nextpage != null){
    urlWithPage = url+nextpage
    command = "wget $urlWithPage -O mapping.txt -o log"
    command.execute().text

    String icd2do = new File('mapping.txt').text
    mappingDict = jsonSlurper.parseText(icd2do)
    nextpage = (mappingDict.nextPage == null)? null:'&page='+mappingDict.nextPage
    mappingDict.collection.each{
        println(it.classes[0]['@id']+"\t"+it.classes[1]['@id'])
    }
}