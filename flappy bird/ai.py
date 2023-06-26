import random
import pygame
import numpy
import math
import time
import copy
import os
pygame.init()

def text_objects(text, font, col):
    textSurface = font.render(text, True, col)
    return textSurface

def write(text,col,size):
    largeText = pygame.font.SysFont("impact", size)
    TextSurf = text_objects(text, largeText, col)
    return TextSurf

def sigmoid(num):
    try:
        return (1/(1+(2.71828**(-(num)))))
    except:
        return 1

def reversesigmoid(num):
    try:
        return math.log(num/(1-num),2.71828)
    except:
        if num>0.5:
            num = 0.9999999999
        else:
            num = 0.0000000001
        return math.log(num/(1-num),2.71828)

def integratedsigmoid(num):
    #print(num.shape)
    num = num.tolist()
    if len(num) == 1:
        num = num[0]
    else:
        num = [a for a in range(len(num))]
    postmaths = []
    for a in range(len(num)):
        postmaths.append([(1/(1+(2.71828**(-(num[a])))))*(((-2.71828**(-(num[a]))))/(1+(2.71828**(-(num[a])))))])
    return numpy.matrix(postmaths)

def softmax(out):
    maxed = [0 for a in range(len(out))]
    tot = sum([2.71828**(b) for b in out])
    for a in range(len(out)):
        maxed[a] = (2.71828**out[a])/tot
    return maxed

class Node:
    def __init__(self,nodenum,nodetotal,displayx,displayy,nodesprevlayer):
        self.nodenum = nodenum
        self.displayx = displayx
        self.displayy = displayy
        self.nodesprevlayer = nodesprevlayer
        #if self.nodesprevlayer!=0:
        self.bias = (random.random()-0.5)*2
        self.connectionweights = [(random.random()-0.5)*2 for x in range(self.nodesprevlayer)]
        self.connectionweightsdisplay = [1/(1+(2.71828**(-self.connectionweights[x]))) for x in range(len(self.connectionweights))]
        self.activation = 0
        self.activationz = 0
    def process(self,connections):
        self.activationz = 0
        for a in range(len(self.connectionweights)):
            #print(self.connectionweights[a],connections[a].activation)
            self.activationz+=float(self.connectionweights[a])*float(connections[a].activation)
        self.activationz+=float(self.bias)
        self.activation = sigmoid(self.activationz)
    def edit(self,mutationchance):
        if self.nodesprevlayer!=0:
            if random.random()<mutationchance: self.bias = random.gauss(self.bias,0.1) 
            for a in range(len(self.connectionweights)):
                if random.random()<mutationchance:
                    #self.connectionweights[a] = (random.random()-0.5)*2
                    self.connectionweights[a] = random.gauss(self.connectionweights[a],0.1) 
    def mixgenes(self,parentnode):
        if random.random()>0.5:
            self.bias = parentnode.bias
        for a in range(len(self.connectionweights)):
            if random.random()>0.5:
                self.connectionweights[a] = parentnode.connectionweights[a]

 

class Layer:
    def __init__(self,layernum,nodenum,displayx,displayheight,nodesprevlayer):
        self.layernum = layernum
        self.nodenum = nodenum
        self.displayx = displayx
        self.displayheight = displayheight
        self.nodesprevlayer = nodesprevlayer
        self.nodes = [Node(x,nodenum,displayx,int(displayheight/(nodenum+1)*(x+1)),self.nodesprevlayer) for x in range(self.nodenum)]
    def editnodes(self,mutationchance):
        for a in range(len(self.nodes)):
            self.nodes[a].edit(mutationchance)
    def mixgenes(self,parentslayer):
        for a in range(len(self.nodes)):
            self.nodes[a].mixgenes(parentslayer.nodes[a])

class AI:
    def __init__(self,displayx,displayy,displaywidth,displayheight,nodesperlayer,inputnodes,outputnodes,hiddenlayertotal,pullfrom):
        self.hiddenlayertotal = hiddenlayertotal
        self.nodesperlayer = nodesperlayer
        self.outputnodes = outputnodes
        self.inputnodes = inputnodes
        self.fitness = 0
        self.setdisplay(displayx,displayy,displaywidth,displayheight)
        self.layers = [Layer(0,self.inputnodes,0,self.displayheight,0)]
        for a in range(self.hiddenlayertotal):
            self.layers.append(Layer(a+1,self.nodesperlayer,int(self.displaywidth/(self.hiddenlayertotal+1)*(a+1)),self.displayheight,len(self.layers[-1].nodes)))
        self.layers.append(Layer(self.hiddenlayertotal+1,self.outputnodes,self.displaywidth,self.displayheight,len(self.layers[-1].nodes)))
        if pullfrom!='none':
            self.readnet(pullfrom,'none')
    def makematrix(self):
        self.layerconmatrix = [numpy.matrix([[self.layers[x+1].nodes[y].connectionweights[z] for z in range(len(self.layers[x+1].nodes[y].connectionweights))] for y in range(len(self.layers[x+1].nodes))]) for x in range(len(self.layers)-1)]
        self.nodematrix = [numpy.matrix([self.layers[x].nodes[y].activation for y in range(len(self.layers[x].nodes))]) for x in range(len(self.layers))]

    def processinput(self,image):
        self.makematrix()
        for nod in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[nod].activation = image[nod]
        self.nodematrix[0] = numpy.matrix([image[y] for y in range(self.inputnodes)])
        for lay in range(len(self.nodematrix)-1):
            self.nodematrix[lay] = numpy.matrix([self.layers[lay].nodes[y].activation for y in range(len(self.layers[lay].nodes))])
##            print('activations',lay,self.nodematrix[lay])
##            print('connections',lay,self.layerconmatrix[lay])
##            print('multiplicas',lay,numpy.multiply(self.nodematrix[lay],self.layerconmatrix[lay]))
            self.nodematrix[lay+1] = numpy.multiply(self.nodematrix[lay],self.layerconmatrix[lay])
            self.nodematrix[lay+1] = self.nodematrix[lay+1].sum(1)
##            print('additioneds',lay,self.nodematrix[lay+1])
            self.nodematrix[lay+1] = self.nodematrix[lay+1].reshape(self.nodematrix[lay+1].shape[1],self.nodematrix[lay+1].shape[0])
            for a in range(len(self.layers[lay+1].nodes)):
                self.layers[lay+1].nodes[a].activationz = self.nodematrix[lay+1].item((a))+float(self.layers[lay+1].nodes[a].bias)
                self.layers[lay+1].nodes[a].activation = sigmoid(float(self.layers[lay+1].nodes[a].activationz))
##        print([self.layers[-1].nodes[x].activation for x in range(len(self.layers[-1].nodes))])
        output = [self.layers[-1].nodes[x].activation for x in range(len(self.layers[-1].nodes))]
        return output
##        for a in range(len(output)):
##            self.layers[-1].nodes[a].activation = output[a]


    def processinputold(self,image):
        for nod in range(len(self.layers[0].nodes)):
            self.layers[0].nodes[nod].activation = image[nod]
        for lay in range(len(self.layers)-1):
            for nod in range(len(self.layers[lay+1].nodes)):
                self.layers[lay+1].nodes[nod].process(self.layers[lay].nodes)

    def fitcalc(self,answer):
        goal = answer
        goal[int(answer)] = 1
        self.wantedchanges = [numpy.matrix([0.0 for a in range(len(self.layers[b].nodes))]) for b in range(len(self.layers))]
        self.wantedchanges[0] = []
        self.amatrix = [numpy.matrix([self.layers[lay].nodes[a].activation for a in range(len(self.layers[lay].nodes))]) for lay in range(len(self.layers))]
        self.zmatrix = [numpy.matrix([self.layers[lay].nodes[a].activationz for a in range(len(self.layers[lay].nodes))]) for lay in range(len(self.layers))]
        self.wmatrix = [numpy.matrix([[self.layers[lay].nodes[a].connectionweights[w] for w in range(len(self.layers[lay].nodes[a].connectionweights))] for a in range(len(self.layers[lay].nodes))]) for lay in range(len(self.layers))]           

        self.wantedchanges[-1] = numpy.matrix([output[c]-goal[c] for c in range(len(self.layers[-1].nodes))])

        for a in range(len(goal)):
            self.fitness += ((output[a]-goal[a])**2)
            
    def setdisplay(self,displayx,displayy,displaywidth,displayheight):
        self.displayx = displayx
        self.displayy = displayy
        self.displaywidth = displaywidth
        self.displayheight = displayheight
        
    def begincycle(self):
        self.fitness = 0
        self.wantedchanges = [numpy.matrix([0.0 for a in range(len(self.layers[b].nodes))]) for b in range(len(self.layers))]
        self.wantedchanges[0] = []
        self.wcbias = [numpy.matrix([0.0 for a in range(len(self.layers[b].nodes))]) for b in range(len(self.layers))]
        self.wcbias[0] = []
        self.wcweights = [numpy.matrix([[0.0 for c in range(len(self.layers[b].nodes[a].connectionweights))] for a in range(len(self.layers[b].nodes))]) for b in range(1,len(self.layers))]
        self.wcweights.insert(0,[])
##        print('activations',self.amatrix)
##        print('active z',self.zmatrix)
##        print('weights',self.wmatrix)
##        print('biases',[numpy.matrix([self.layers[lay].nodes[a].bias for a in range(len(self.layers[lay].nodes))]) for lay in range(len(self.layers))])
        
    def backpropagation(self,image,lay,answer):
        goal = answer
        if lay != len(self.layers)-1:
            self.wantedchanges[lay] -= (numpy.multiply(self.wmatrix[lay+1].transpose().dot(self.wantedchanges[lay+1].transpose()),integratedsigmoid(self.zmatrix[lay]))).transpose()
        self.wcweights[lay] += self.wantedchanges[lay].transpose().dot(self.amatrix[lay-1])
        self.wcbias[lay] += numpy.sum(self.wantedchanges[lay])      
        if lay!=1:
            self.backpropagation(image,lay-1,answer)
           
    def gradientdescent(self,loops):
        self.wcbiaslist = []
        for a in range(len(self.wcbias)):
            try:self.wcbiaslist.append(self.wcbias[a].tolist())
            except:self.wcbiaslist.append([])
        self.wcweightslist = []
        for a in range(len(self.wcweights)):
            try:self.wcweightslist.append(self.wcweights[a].tolist())
            except:self.wcweightslist.append([])
        #print(self.wcweightslist[2])
##        print('bias changes',self.wcbiaslist)
##        print('weights changes',self.wcweightslist[2])
##        print('changes',self.wantedchanges)
        #print(self.wcbiaslist)
        for lay in range(len(self.wcweightslist)-1):
            for nod in range(len(self.wcweightslist[lay+1])):
                self.layers[lay+1].nodes[nod].bias -= (self.wcbiaslist[lay+1][0][nod]/loops)*0.2
                for con in range(len(self.wcweightslist[lay+1][nod])):
                    self.layers[lay+1].nodes[nod].connectionweights[con] -= (self.wcweightslist[lay+1][nod][con]/loops)*0.2
##                    if lay == 1:
##                        print(self.layers[lay+1].nodes[nod].connectionweights[con],self.layers[lay+1].nodes[nod].bias,nod,con)
                    



 
    def displaynetwork(self,screen,connectcompress):
        if connectcompress[3]:
            for a in range(len(self.layers)):
                #print(self.nodematrix)
                for b in range(len(self.layers[a].nodes)):
                    self.layers[a].nodes[b].activation = self.nodematrix[a].item((b))
        for lay in range(len(self.layers)):
            compressinc = -1
            for nod in range(len(self.layers[lay].nodes)):
                compressinc+=1
                if lay > 0:
                    section = 1
                    if lay==1:section = 0
                    elif lay == len(self.layers)-1:section = 2
                    if connectcompress[section]:
                        inc = int(math.sqrt(len(self.layers[lay-1].nodes)))
                        for con in range(0,len(self.layers[lay-1].nodes),inc):
                            try: pygame.draw.line(screen,(255-(self.layers[lay].nodes[nod].connectionweightsdisplay[con+compressinc]*255),50,0),(self.layers[lay].nodes[nod].displayx+self.displayx,self.layers[lay].nodes[nod].displayy+self.displayy),(self.layers[lay-1].nodes[con+compressinc].displayx+self.displayx,self.layers[lay-1].nodes[con+compressinc].displayy+self.displayy),1)
                            except: pass
                            if compressinc >= inc:
                                compressinc = -1
                    else:
                        for con in range(len(self.layers[lay-1].nodes)):
                            pygame.draw.line(screen,(255-(self.layers[lay].nodes[nod].connectionweightsdisplay[con]*255),50,0),(self.layers[lay].nodes[nod].displayx+self.displayx,self.layers[lay].nodes[nod].displayy+self.displayy),(self.layers[lay-1].nodes[con].displayx+self.displayx,self.layers[lay-1].nodes[con].displayy+self.displayy),1)
                    for con in range(len(self.layers[lay-1].nodes)):
                        prevlaynodecol = max([min([self.layers[lay-1].nodes[con].activation,1]),0])
                        if nod == len(self.layers[lay].nodes)-1:
                            pygame.draw.circle(screen,(prevlaynodecol*255,prevlaynodecol*255,prevlaynodecol*255),(self.layers[lay-1].nodes[con].displayx+self.displayx,self.layers[lay-1].nodes[con].displayy+self.displayy),5)
                    if section == 2:pygame.draw.circle(screen,(self.layers[lay].nodes[nod].activation*255,self.layers[lay].nodes[nod].activation*255,self.layers[lay].nodes[nod].activation*255),(self.layers[lay].nodes[nod].displayx+self.displayx,self.layers[lay].nodes[nod].displayy+self.displayy),5)

    def evolve(self,childnumber,mutationchance,parent):
        children = [copy.deepcopy(self) for a in range(childnumber)]
        for a in range(len(children)):
            for b in range(len(children[a].layers)):
                children[a].layers[b].mixgenes(parent.layers[b])
                children[a].layers[b].editnodes(mutationchance)
        return children
    def evolvesingle(self,childnumber,mutationchance):
        children = [copy.deepcopy(self) for a in range(childnumber)]
        for a in range(len(children)):
            for b in range(len(children[a].layers)):
                children[a].layers[b].editnodes(mutationchance)
        return children

    def storenet(self,name):
        with open(name+'.txt','w') as f:
            for a in range(len(self.layers)):
                f.write(str(len(self.layers[a].nodes))+';')
            f.write('\n')
            for lay in range(len(self.layers)-1):
                for nod in range(len(self.layers[lay+1].nodes)):
                    f.write(str(self.layers[lay+1].nodes[nod].bias)+';')
                    for con in range(len(self.layers[lay+1].nodes[nod].connectionweights)):
                        f.write(str(self.layers[lay+1].nodes[nod].connectionweights[con])+';')
                    f.write('\n')
                    
    def readnet(self,name,filepath):
        if filepath == 'none':
            path = os.path.abspath(os.getcwd())+'\\' + name + '.txt'
        else:
            path = os.path.abspath(os.getcwd())+'\\'+filepath+'\\' + name + '.txt'
        with open(path,'r') as f:
            fil = f.readlines()
            data = [fil[x].split(';') for x in range(len(fil))]
            for a in range(len(data)):
                del data[a][-1]
##            print(data)
            net = AI(self.displayx,self.displayy,self.displaywidth,self.displayheight,int(data[0][1]),int(data[0][0]),int(data[0][-1]),len(data[0])-2,'none')

            self.layers = [Layer(0,int(data[0][0]),0,self.displayheight,0)]
            for a in range(self.hiddenlayertotal):
                self.layers.append(Layer(a+1,int(data[0][1]),int(self.displaywidth/(self.hiddenlayertotal+1)*(a+1)),self.displayheight,len(self.layers[-1].nodes)))
            self.layers.append(Layer(len(data[0])-1,int(data[0][-1]),self.displaywidth,self.displayheight,len(self.layers[-1].nodes)))
        
            inc = 1
            for lay in range(len(self.layers)-1):
                for nod in range(len(self.layers[lay+1].nodes)):
                    self.layers[lay+1].nodes[nod].bias = float(data[inc][0])
                    for con in range(len(self.layers[lay+1].nodes[nod].connectionweights)):
                        self.layers[lay+1].nodes[nod].connectionweights[con] = float(data[inc][con+1])
                        self.layers[lay+1].nodes[nod].connectionweightsdisplay[con] = 1/(1+(2.71828**(-float(data[inc][con+1]))))
                    inc+=1
            net.makematrix()
        self.rednet = net
        #self = net



##class MAIN:
##    def __init__(self,hiddenlayertotal,nodesperlayer,inputnodes,outputnodes,aix,aiy,aiwidth,aiheight,screenwidth,screenheight,connectcompress):
##        self.hiddenlayertotal = hiddenlayertotal
##        self.nodesperlayer = nodesperlayer
##        self.outputnodes = outputnodes
##        self.inputnodes = inputnodes
##        self.aix = aix
##        self.aiy = aiy
##        self.aiwidth = aiwidth
##        self.aiheight = aiheight
##        self.screenwidth = screenwidth
##        self.screenheight = screenheight
##        self.connectcompress = connectcompress
##        self.clock = pygame.time.Clock()
##        self.screen = pygame.display.set_mode((screenwidth,screenheight))
##        self.done = False
##        self.generationnum = 0
##        self.traininc = 0
##        self.clonenum = 2
##        self.aistraining = 1
##        self.data = [self.getimage(a) for a in range(60000)]
##        #self.ai = [self.readnetzeros('gen 8611 ai num '+str(a),'first gen trained ai')for a in range(self.aistraining)]
##        self.ai = [self.readnet('4x32 ai gd gen 24883','trained ais\\4x32 working ai')for a in range(self.aistraining)]
##        #self.ai = [AI(aix,aiy,aiwidth,aiheight,nodesperlayer,inputnodes,outputnodes,hiddenlayertotal) for x in range(self.aistraining)]
##        for a in self.ai:
##            a.makematrix()
##
##    def main(self):
##        while not self.done:
##            self.darwinism()
##            self.dloop()
##        pygame.QUIT()
##    def dloop(self):
##        for event in pygame.event.get():
##            if event.type == pygame.QUIT:
##                self.done = True
##        self.screen.fill((0,0,70))
##        self.ai[0].displaynetwork(self.screen,self.connectcompress)
##        self.displayimage(self.screenwidth-280,0,10,self.image)
##        pygame.display.flip()
##        self.clock.tick(1000)
## 
##    def darwinism(self):
##        loops = 1000
##        print('------------------------------ gen number '+str(self.generationnum)+' ------------------------------')
##        for a in self.ai:
##            a.begincycle()
##        for imnum in range(self.traininc,loops+self.traininc):
##            #self.image,self.answer = self.getimage(imnum)
##            self.image = self.data[imnum][0]
##            self.answer = self.data[imnum][1]
####            thing = [[[0,0],0],[[0,1],1],[[1,0],2],[[1,1],3]]
####            ran = random.randint(0,3)
####            self.answer = thing[ran][1]
####            self.image = thing[ran][0]
##            for b in range(len(self.ai)):
##                self.ai[b].processinputmatrix(self.image)
##                self.ai[b].fitcalc(self.answer)
##                self.ai[b].backpropagation(self.image,len(self.ai[b].layers)-1,self.answer)
##            if imnum%int(loops/10) == 0:
##                print(str(int((imnum-self.traininc)/loops*100)+1)+'%',end=' ')
##                self.dloop()
##        self.generationnum+=1
##        for a in self.ai:
##            a.gradientdescent(loops)
##            a.fitness/=loops
##        print([self.ai[x].fitness for x in range(self.aistraining)])
##        self.traininc+=loops
##        if self.traininc > 59999:
##            self.traininc = 0
##            
##    def cloneai(self,index):
##        self.storenet(self.ai[index],'cloneing')
##        children = [self.readnet('cloneing','AI project') for a in range(self.clonenum)]
##        return children
##    
##    def storenet(self,net,name):
##        with open(name+'.txt','w') as f:
##            for a in range(len(net.layers)):
##                f.write(str(len(net.layers[a].nodes))+';')
##            f.write('\n')
##            for lay in range(len(net.layers)-1):
##                for nod in range(len(net.layers[lay+1].nodes)):
##                    f.write(str(net.layers[lay+1].nodes[nod].bias)+';')
##                    for con in range(len(net.layers[lay+1].nodes[nod].connectionweights)):
##                        f.write(str(net.layers[lay+1].nodes[nod].connectionweights[con])+';')
##                    f.write('\n')
##                    
####    def storenetdis(self,net,name):
####        with open(name+'.txt','w') as f:
####            for a in range(len(net.layers)):
####                f.write(str(len(net.layers[a].nodes))+';')
####            f.write('\n')
####            for lay in range(len(net.layers)-1):
####                for nod in range(len(net.layers[lay+1].nodes)):
####                    f.write(str(net.layers[lay+1].nodes[nod].activation)+';')
####                    f.write(str(net.layers[lay+1].nodes[nod].bias)+';')
####                    for con in range(len(net.layers[lay+1].nodes[nod].connectionweights)):
####                        f.write(str(net.layers[lay+1].nodes[nod].connectionweights[con])+';')
####                    f.write('\n------\n')
##
##    def readnet(self,name,filepath):
##        if filepath == 'idontexist':
##            path = os.path.abspath(os.getcwd())+'\\' + name + '.txt'
##        else:
##            path = os.path.abspath(os.getcwd())+'\\'+filepath+'\\' + name + '.txt'
##        with open(path,'r') as f:
##            fil = f.readlines()
##            data = [fil[x].split(';') for x in range(len(fil))]
##            for a in range(len(data)):
##                del data[a][-1]
####            print(data)
##            net = AI(self.aix,self.aiy,self.aiwidth,self.aiheight,int(data[0][1]),int(data[0][0]),int(data[0][-1]),len(data[0])-2)
##            inc = 1
##            for lay in range(len(net.layers)-1):
##                for nod in range(len(net.layers[lay+1].nodes)):
##                    net.layers[lay+1].nodes[nod].bias = float(data[inc][0])
##                    for con in range(len(net.layers[lay+1].nodes[nod].connectionweights)):
##                        net.layers[lay+1].nodes[nod].connectionweights[con] = float(data[inc][con+1])
##                        net.layers[lay+1].nodes[nod].connectionweightsdisplay[con] = 1/(1+(2.71828**(-net.layers[lay+1].nodes[nod].connectionweights[con])))
##                    inc+=1
##            net.makematrix()
##        return net
##    
##    def readnetzeros(self,name,filepath):
##        if filepath == 'idontexist':
##            path = os.path.abspath(os.getcwd())+'\\' + name + '.txt'
##        else:
##            path = os.path.abspath(os.getcwd())+'\\'+filepath+'\\' + name + '.txt'
##        with open(path,'r') as f:
##            fil = f.readlines()
##            data = [fil[x].split(';') for x in range(len(fil))]
##            for a in range(len(data)):
##                del data[a][-1]
##            net = AI(self.aix,self.aiy,self.aiwidth,self.aiheight,int(data[0][1]),int(data[0][0]),int(data[0][3]),len(data[0])-2)
##            inc = 1
##            for lay in range(len(net.layers)-1):
##                for nod in range(len(net.layers[lay+1].nodes)):
##                    net.layers[lay+1].nodes[nod].bias = float(0.1)
##                    for con in range(len(net.layers[lay+1].nodes[nod].connectionweights)):
##                        net.layers[lay+1].nodes[nod].connectionweights[con] = float(0.1)
##                    inc+=1
##            net.makematrix()
##        return net
##    
##    def getimage(self,num):
##        if num%1000 == 0:
##            print(num)
##        with open(os.path.abspath(os.getcwd())+'\\training data\\' + str(num) + '.txt') as f:
##            text = str(f.read())
##        items = text.split(';')
##        answer = items[-1]
##        del items[-1]
##        image = [float(items[a]) for a in range(len(items))]
##        return image,answer
##
##    def displayimage(self,x,y,scale,image):
##        for a in range(28):
##            for b in range(28):
##                pygame.draw.rect(self.screen,(image[b*28+a]*255,image[b*28+a]*255,image[b*28+a]*255),pygame.Rect(x+a*scale,y+b*scale,scale,scale))

###main = MAIN(1,8,2,4,10,10,600,480,900,500,[1,0,0,0])
##main = MAIN(12,69,784,10,10,10,1500,980,1800,1000,[1,0,0,0])
##main.main()
