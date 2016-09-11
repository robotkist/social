import numpy as np
from sets import Set
import time
import types


def random_selection1(number_of_choices,discrete_distribution): #return the location of one got selected
    #maybe add assert here later on
    tmp=np.random.choice(np.arange(1,number_of_choices+1),p=discrete_distribution)
    return tmp
def random_selection2(discrete_distribution): #return the selected prob.
    number_of_choices=len(discrete_distribution)
    tmp=np.random.choice(np.arange(number_of_choices),p=discrete_distribution)
    return p[tmp]
def get_discrete_distribution(list_of_not_yet_normalized_numbers):
    tmp=np.asarray(list_of_not_yet_normalized_numbers)
    summed=np.sum(tmp)
    return tmp/summed
    
def knowledge_to_job(knowledge):
    tmp=knowledge
    tmp.__class__=Job
    tmp._init_Job()
    return tmp

class Granules(object): #knowledge gradule with actual contents. leaf node Knowledge will be granules
    def __init__(self):
        self.dict=dict() #dictionary of its elemental properties and values
        self.set=set() #set of children
        self.name=None
    def build(self,dictionary):
        self.dict=dictionary
        tmp=self.dict.keys()
        self.set=set(tmp)
    def expand(self,terms_to_expand,libraries_to_search): # takes lists
        additional_dictionary=dict()
        for library in libraries_to_search
            tmp,_=self.load_from_library(library,terms_to_expand)
            additional_dictionary.update(tmp)
        self.dict.update(additional_dictionary)
        keys_to_return=self.dict.keys()
        self.set.update(keys_to_return)
        return len(additional_dictionary)
    def load_from_library(self,library,list_of_keys): #extract sub dictionary
        tmp=dict((k, library[k]) for k in list_of_keys if k in library)
        length=len(tmp)
        return tmp, length


class Knowledge(Granules):
    '''
    Knowledge has parents(higher level concepts), children(lower level concepts).
    Siblings share parents.
    activation history holds 'pointer' to past n parents and past n children
    '''
    def __init__(self,name,default_library,input_channels):
        super(Knowledge,self).__init__()
        self.parents=dict() #name of parents. dictionary of (name of parent, connection strength). content can be found with gradule
        self.siblings=dict()
        self.children=dict()
        self.forward_history=[]
        self.backward_history=[]
        self.name=name
        self.forward=None #name of most recently activated parent
        self.backward=None #name of most recently activated children
        self.is_leaf=False
        self.is_root=False
        self.library=default_library
        self.list_of_inputs=input_channels
        
    def reset_history(self):
        self.forward_history=[]
        self.backward_history=[]
    def get_parent(self):
        return self.parents
    def get_siblings(self):
        return self.siblings
    def get_children(self):
        return self.children
    def get_name(self):
        return self.name
    def run(self, self.list_of_inputs): #always give run a list
        #extract arguments from list of inputs from a dictionary, namely InputChannel
        return 0

    def forward(self,parent_name): #forward chain. runs Knowledge.run in cascade and list of selected forward chain
        if !self.is_leaf
            self.forward=parent_name
            self.forward_history+=self.forward
            self.run(argument_list)
            child_name=self.select_child()
            child_instance=self.library.extract_knowledge(child_name)
            child_instance.forward(self.name,child_instance.list_of_inputs)
            print self.forward
        return self.forward
    def backward(self,child_name): #backward chain. results in list of selected Knowledges in backward chain
        if !self.is_root
            self.backward=child_name
            self.backward_history+=self.backward
            self.run(self.list_of_inputs)
            parent_name=self.select_parent()
            parent_instance=self.library.extract_knowledge(parent_name)
            parent_instance.backward(self.name,parent_instance.list_of_inputs)
            #list(set(x) - set(y))
            print self.backward
        return self.backward
    def select_parent(self):
        tempv=self.parents.values()
        prob_distribution=get_discrete_distribution(tmpv)
        selectedv=random_selection2(prob_distribution)
        for key,values in self.parents:
            if selectedv==values:
                return key
            else print 'something is fucking wrong'
    def select_child(self):
        tempv=self.children.values()
        prob_distribution=get_discrete_distribution(tmpv)
        selectedv=random_selection2(prob_distribution)
        for key,values in self.children:
            if selectedv==values:
                return key
            else print 'something is fucking wrong'




    def select_child(self,child)



class Job(Knowledge):
    '''
    Job contains list of Knowledge(children) and their priority.
    A Job itself is a Knowledge, so it must be provided with name when instansiated
    '''
    def __init__(self):
        super(Job,self).__init__()
        self._init_Job()
    def _init_Job(self):
        self.knowledge_list=[]
        self.success_rates=[]
        self.fail_rates=[]
        self.job_list_success=dict()
        self.job_list_fail=dict()
        self.update_rate=0.1
    def reset_success_rates(self):
        self.success_rates=np.zeros((1,len(self.knowledge_list)))
    def reset_fail_rates(self):
        self.fail_rates=np.zeros((1,len(self.knowledge_list)))
    def reset_rates(self):
        self.reset_success_rates(self)
        self.reset_fail_rates
    def create_job_list(self):
        self.job_list_success=dict(zip(self.knowledge_list,self.success_rates))
        self.job_list_fail=dict(zip(self.knowledge_list,self.fail_rates))
    def adjust_priority(self):
        tempkeys=self.children.keys()
        for keys in tempkeys
            self.children[keys]=self.children[keys]+self.update_rate*(self.job_list_success[keys]-self.job_list_fail[keys])



        
class Library(object): #dictionary with Knowledge.name as keys and actual instanciation of the Knowledge
    def __init__(self):
        self.content=dict()
        self.stored_place=None #cloud, laptop, or tablet
        self.name=None
    def build(self,list_of_keys,list_of_values):
        tmp=zip(list_of_keys,list_of_values)
        self.content=dict(tmp)
    def push_to_library(self,target_library):
        target_library.content.update(self.content)
    def pull_from_library(self,library,list_of_keys): #extract sub dictionary
        tmp=dict((k, library[k]) for k in list_of_keys if k in library)
        length=len(tmp)
        self.content.update(tmp)
        return length
    def extract_knowledge(self,knowledge_name):
        return self.content.get(knowledge_name)
        
def return_x(argument_list):
    return argument_list[0],argument_list[1]
rx=return_x
a=Knowledge('bullshit')
print a.run([])
a.run=rx
#a.run=types.MethodType(rx,a)
d=(1,2)
print a.run(d)
start= time.time()

end=time.time()


vid1=np.array([[0,0,0,1],[0,0,1,0],[0,0,0,0],[0,0,0,0]])

Lib=Library()

operate=Knowledge('Operate',Lib,vid1)
engage=Knowledge('Engage',Lib,vid1)
dem=Knowledge('Detect_Emotion',Lib,vid1)
aprc=Knowledge('Approach',Lib,vid1)

def detect_emotion(video_input):
    for rows in video_input:
        if k[3]==1
            return 'happy'
    return 'failed to detect emotion'
dem.run=detect_emotion



'''
casting example (for knowledge->job)

class A(object):
    def __init__(self):
        self.x = 1

class B(A):
    def __init__(self):
        super(B, self).__init__()
        self._init_B()
    def _init_B(self):
        self.x += 1

a = A()
b = a
b.__class__ = B
b._init_B()


'''