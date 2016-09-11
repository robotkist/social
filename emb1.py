import numpy as np
from sets import Set
import time
import types
import matplotlib.pyplot as plt



def random_selection1(number_of_choices,discrete_distribution): #return the location of one got selected
    #maybe add assert here later on
    tmp=np.random.choice(np.arange(1,number_of_choices+1),p=discrete_distribution)
    return tmp.astype(np.float32)
def random_selection2(discrete_distribution): #return the selected prob.
    number_of_choices=len(discrete_distribution)
    tmp=np.random.choice(np.arange(number_of_choices),p=discrete_distribution)
    
    return np.float32(discrete_distribution[tmp])
def get_discrete_distribution(list_of_not_yet_normalized_numbers):
    tmp=np.asarray(list_of_not_yet_normalized_numbers)
    tmp=tmp.astype(np.float32)
    summed=np.sum(tmp)
    summed=np.float32(summed)
    return tmp/summed
    
def knowledge_to_job(knowledge):
    tmp=knowledge
    tmp.__class__=Job
    tmp._init_Job()
    #print tmp.name
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
        for library in libraries_to_search:
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
        #print self.name
        
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
    def run(self, list_of_inputs): #always give run a list
        print 'run %s' % self.name
       #print self.name
        #extract arguments from list of inputs from a dictionary, namely InputChannel
        return 0

    def _forward(self,parent_name): #forward chain. runs Knowledge.run in cascade and list of selected forward chain
        print 'forward %s' % self.name
        #print self.name
        
        if self.is_leaf==False:
            self.forward=parent_name
            
            run_result=self.run(self.list_of_inputs)
            child_name=self.select_child()
            child_instance=self.library.extract_knowledge(child_name)
            self.forward_history.append(child_instance.name)
            child_instance._forward(self.name)
            print 'self.forward'
            print self.forward
        else:
            self.run(self.list_of_inputs)
        print 'returning self.forward for %s' % self.name
        return self.forward
        
        
    def _backward(self,child_name): #backward chain. results in list of selected Knowledges in backward chain
        print 'backward %s' % self.name
        #print self.name
        if self.is_root==False:
            self.backward=child_name
            
            #self.run(self.list_of_inputs)
            parent_name=self.select_parent()
            parent_instance=self.library.extract_knowledge(parent_name)
            parent_instance._backward(self.name)
            #list(set(x) - set(y))
            print self.backward
        print 'returning self.backward %s' % self.name
        return self.backward
    def select_parent(self):
        tempv=self.parents.values()
        prob_distribution=get_discrete_distribution(tempv)
        selectedv=random_selection2(prob_distribution)
        tmp=tempv
        tmp=np.asarray(tmp,dtype=np.float32)
        summed=np.sum(tmp)
        summed=np.float32(summed)
        selectedv=selectedv*summed
        
        for key,values in self.parents.iteritems():
            if selectedv==values:
                return key
           
    def select_child(self):
        tempv=self.children.values()
        prob_distribution=get_discrete_distribution(tempv)
        selectedv=random_selection2(prob_distribution)
        print self.children
        print selectedv
        tmp=tempv
        tmp=np.asarray(tmp,dtype=np.float32)
        summed=np.sum(tmp)
        summed=np.float32(summed)
        selectedv=selectedv*summed
        for key, values in self.children.iteritems():
            if selectedv==values:
                return key
           







class Job(Knowledge):
    '''
    Job contains list of Knowledge(children) and their priority.
    A Job itself is a Knowledge, so it must be provided with name when instansiated
    '''
    def __init__(self):
        super(Job,self).__init__()
        self._init_Job()
    def _init_Job(self):
        self.max_ttl=100 #latest global time of completion allowed for a job
        self.knowledge_list=self.children.keys()
        self.success_rates=np.zeros((1,len(self.knowledge_list)),dtype=np.float32)
        self.fail_rates=np.zeros((1,len(self.knowledge_list)),dtype=np.float32)
        self.job_list_success=dict()
        self.job_list_fail=dict()
        self.ttl_list=dict()
        self.completion_list=dict()
        self.update_rate=0.1
        self.completion_rate=np.zeros((1,len(self.knowledge_list)),dtype=np.float32) # 1 being completed, 0 being none done
        self.start_time=np.zeros((1,len(self.knowledge_list)),dtype=np.float32) # when did it start
        self.ttl=self.max_ttl-self.start_time
        self.create_job_list()
        self.job_priorities=self.children
        #print 'init children %s' % self.children
    def reset_success_rates(self):
        self.success_rates=np.zeros((1,len(self.knowledge_list)))
    def reset_fail_rates(self):
        self.fail_rates=np.zeros((1,len(self.knowledge_list)))
    def reset_rates(self):
        self.reset_success_rates(self)
        self.reset_fail_rates
    def create_job_list(self):
        #print 'kn list %s' % self.knowledge_list
        self.job_list_success=dict(zip(self.knowledge_list,self.success_rates[0]))
        #print 'jls %s' % self.job_list_success
        self.job_list_fail=dict(zip(self.knowledge_list,self.fail_rates[0]))
        self.ttl_list=dict(zip(self.knowledge_list,self.ttl[0]))
        self.completion_list=dict(zip(self.knowledge_list,self.completion_rate[0]))
    def adjust_priority(self):
        tempkeys=self.children.keys()
        for keys in tempkeys:
            print 'keys %s' % keys
            #print 'chilsren is %s' % self.children
            print self.children[keys],self.job_list_success[keys],self.job_list_fail[keys],self.completion_list[keys],self.ttl_list[keys]
            tmp=self.children[keys]+(self.update_rate*(self.job_list_success[keys]-self.job_list_fail[keys]))*(1-self.completion_list[keys])*(1/(self.ttl_list[keys]+0.1))
            self.job_priorities[keys]=tmp.astype(np.float32)
            print 'changed priority for this key is %s' % self.job_priorities[keys]
    def _forward(self,parent_name): #forward chain. runs Knowledge.run in cascade and list of selected forward chain
        print 'forward %s' % self.name
        #print self.name
        
        if self.is_leaf==False:
            self.forward=parent_name
            
            
            run_result=self.run(self.list_of_inputs)
            if run_result==1:
                key=self.name
                parent_instance=self.library.extract_knowledge(self.forward)
                print parent_instance.job_priorities.values()
                print 'parent is %s' % parent_instance.name
                print 'p job list %s' % parent_instance.job_list_success
                
                parent_instance.job_list_success[key]+=1
                parent_instance.adjust_priority()
                print parent_instance.job_priorities.values()

            if run_result==-1:
                key=self.name
                parent_instance=self.library.extract_knowledge(self.forward)
                parent_instance.job_list_fail[key]+=1
                parent_instance.adjust_priority()
            
            child_name=self.select_child()
            print child_name
            child_instance=self.library.extract_knowledge(child_name)
            self.forward_history.append(child_instance.name)
            child_instance._forward(self.name)
            print 'self.forward'
            print self.forward
        else:
            self.run(self.list_of_inputs)
        print 'returning self.forward for %s' % self.name
        return self.forward
        
        
    def _backward(self,child_name): #backward chain. results in list of selected Knowledges in backward chain
        print 'backward %s' % self.name
        #print self.name
        if self.is_root==False:
            self.backward=child_name
            #self.run(self.list_of_inputs)
            parent_name=self.select_parent()
            parent_instance=self.library.extract_knowledge(parent_name)
            parent_instance._backward(self.name)
            #list(set(x) - set(y))
            print self.backward
        print 'returning self.backward %s' % self.name
        return self.backward

    def select_parent(self):
        tempv=self.parents.values()
        prob_distribution=get_discrete_distribution(tempv)
        selectedv=random_selection2(prob_distribution)
        tmp=tempv
        tmp=np.asarray(tmp,dtype=np.float32)
        summed=np.sum(tmp)
        summed=np.float32(summed)
        selectedv=selectedv*summed
        
        for key,values in self.parents.iteritems():
            if selectedv==values:
                return key
           
    def select_child(self):
        tempv=self.job_priorities.values()
        print '%s jpv %s ' % (self.name, self.job_priorities)
        prob_distribution=get_discrete_distribution(tempv)
        selectedv=random_selection2(prob_distribution)
        
        #print 'chosen value is %f' % selectedv
        tmp=tempv
        tmp=np.asarray(tmp,dtype=np.float32)
        summed=np.sum(tmp)
        summed=np.float32(summed)
        selectedv=selectedv*summed

        print 'chosen value is %f' % selectedv
        for key, values in self.job_priorities.iteritems():
            print selectedv,values
            #if np.float32(selectedv)==np.float32(values):
            if abs(np.float32(selectedv)-np.float32(values))<0.00001:
                print 'selcted key is %s' % key
                return key

        
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
'''
class Envir(object): #we need to make class for environment
    def __init__(self):
'''

start= time.time()



vid1=np.array([[0,0,0,1],[0,0,1,0],[0,0,0,0],[0,0,0,0]])
vid2=np.zeros_like(vid1)
vid2[0]=[0,0,0,1]
vid3=np.array([[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,0,1,0]])

Lib=Library()
variable_list=[]

operate=Knowledge('Operate',Lib,vid1)

variable_list.append(operate)
operate.is_root=True
operate.children={'Engage':10.}

engage=Knowledge('Engage',Lib,vid1)
variable_list.append(engage)
engage.parents={'Operate':10.}
engage.children={'Detect_Emotion':2.,'Approach':4.}
dem=Knowledge('Detect_Emotion',Lib,vid1)
variable_list.append(dem)
dem.parents={'Engage':9.}
#dem.is_leaf=True
dem.children={'Listen':9.}
aprc=Knowledge('Approach',Lib,vid1)
variable_list.append(aprc)
aprc.parents={'Engage':5.}
aprc.children={'Listen':5.}
aprc.is_leaf=False
var_name_list=['Operate','Engage','Detect_Emotion','Approach']
Lib.build(var_name_list,variable_list)

listen=Knowledge('Listen',Lib,vid1)
listen.parents={'Detect_Emotion':10.,'Approach':2.}
listen.is_leaf=True


var_name_list.append('Listen')
variable_list.append(listen)
tmpd=dict(zip(var_name_list,variable_list))
Lib.content.update(tmpd)
#print Lib.content



def detect_emotion(video_input):
    for rows in video_input:
        if rows[3]==1:
            print 'happy'
            return 1
    print 'failed to detect emotion'
    return -1
def approach(video_input):
    '''
    for rows in video_input:
        for i in range(video_input.shape[0]-1):
            if (rows[i]!=rows[i+1]).all():
                print 'moved'
                return 1
    '''
    if video_input[0][1]==1:
        print 'moved'
        return 1
    else: 
        print 'cannot move'
        return -1
    
dem.run=detect_emotion
#backup Knowledges
operate_cpy=operate
engage_cpy=engage
dem_cpy=dem
aprc_cpy=aprc

operate=knowledge_to_job(operate)
engage=knowledge_to_job(engage)
dem=knowledge_to_job(dem)
aprc=knowledge_to_job(aprc)
listen=knowledge_to_job(listen)

aprc.run=approach
#engage.run(vid1)
'''
forward_list=[]
backward_list=[]
forward_list+=operate._forward('Concept')
print '\n\n end of forward \n\n'
#print engage.get_parent()
backward_list+=listen._backward('Leaf')
print '\n\nList\n'
print forward_list
print backward_list
'''
'''
Scenario 1
The objective for the robot is to operate.
It will detect emotion or approach. In the beginning approach is easy and more desireable. 
Approach then becomes difficult and fails constantly.
Now the priority shifts to emotion detection, which it does well.

'''
operate.max_ttl=100
#operate.start_time=0
engage.max_ttl=90
#engage.start_time=10
dem.max_ttl=60
#dem.start_time=20
aprc.max_ttl=60
#aprc.start_time=20

for i in range(90):
    print '\ntimestep %d \n' % i
    
    if i>25 and i<35:
        dem.list_of_inputs=vid2
        aprc.list_of_inputs=vid2
    '''
    if i>35:
        dem.list_of_inputs=vid3
        aprc.list_of_inputs=vid3
    '''
    engage._forward('Operate')
    operate.max_ttl-=1
    engage.max_ttl-=1
    dem.max_ttl-=1
    aprc.max_ttl-=1
    listen.max_ttl-=1
    
print engage.forward_history
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