import pygame,math,random,time,copy
import ai
pygame.init()
screen = pygame.display.set_mode((1500, 1000))

def text_objects(text, font, col):
    textSurface = font.render(text, True, col)
    return textSurface, textSurface.get_rect()

def write(x,y,text,col,size):
    largeText = pygame.font.SysFont("calibri",int(size),bold=True)
    TextSurf, TextRect = text_objects(text, largeText, col)
    TextRect.x = int(x)
    TextRect.y = int(y)
    screen.blit(TextSurf, TextRect)

def sigmoid(num,s):
    try:
        return (1/(1+(s**(-(num)))))
    except:
        return 1
    
class BIRB:
    def __init__(self,x,y,aicontrolled,id):
        self.y = y
        self.x = x
        self.yvel = 0
        self.w = 68
        self.h = 48
        if aicontrolled: self.image = pygame.image.load('ai bird.png')
        else: self.image = pygame.image.load('bird.png')
        self.image = pygame.transform.scale(self.image,(self.w,self.h))
        self.image.set_colorkey((255,255,255))
        self.spacedown = False
        self.spacedownused = True
        self.dead = False

        self.blackbox = [0,0]
        self.id = id
        self.aicontrolled = aicontrolled
        
    def control(self,spacedown,walls,diff):
        if self.aicontrolled:
            inpu = self.getaiinputs(walls,diff)
            out = self.net.processinput(inpu)
            self.spacedown = bool(round(out[0]))
        else:
            self.spacedown = spacedown

    def getaiinputs(self,walls,diff):
        inpu = []
        if len(walls)>0:
            b = 0
            while walls[b].x+walls[b].w/2<self.x-self.w/2:
                b+=1
                if b==len(walls):
                    for a in range(3): inpu.append(0)
                    b = -1
            if b!=-1:
                inpu.append(min([(walls[b].x-self.x)/(1500-self.x),1]))
                inpu.append(((walls[b].y+walls[b].gap/2)-180)/602)
                inpu.append(walls[b].gap/350)
                
        else:
            for a in range(3): inpu.append(0)
        inpu.append(max([min([self.y/912,1]),0]))
        inpu.append(sigmoid(self.yvel,1.1))
        #inpu.append(diff*2-1)
        #inpu.append(int(self.spacedownused))
        return inpu
    def move(self):
        if not self.dead:
            self.blackbox[0]+=1
        self.yvel+=1
        self.y+=self.yvel
        if self.spacedown and not self.spacedownused:
            self.spacedownused = True
            self.yvel = -18
        else:
            self.spacedownused = False
    def draw(self,xvel):
        if self.yvel!=0:
            rot = pygame.transform.rotate(self.image,math.atan(-self.yvel/(xvel*2))/math.pi*180)
            rot.set_colorkey((255,255,255))
            screen.blit(rot,(self.x-rot.get_width()/2,self.y-rot.get_height()/2))
        else:
            screen.blit(self.image,(self.x-self.w/2,self.y-self.h/2))
##        if self.aicontrolled:
##            self.net.displaynetwork(screen,(0,0,0,0))
    def die(self):
        self.dead = True
        #print('bird '+str(self.id)+' died at '+str(self.blackbox[0])+' with a score of '+str(self.blackbox[1]))


class wall:
    def __init__(self,diff):
        self.x = 1600
        self.gap = int(random.randint(200,350)*diff)
        self.y = random.randint(80,832-self.gap)
        self.givenscore = False
        
        self.topimage = pygame.image.load('pillar.png')
        self.bottomimage = pygame.transform.flip(self.topimage,False,True)
        self.topimage.set_colorkey((255,255,255))
        self.bottomimage.set_colorkey((255,255,255))
        
        self.w = self.topimage.get_width()
        self.h = self.topimage.get_height()

    def move(self,vel):
        self.x-=vel
    def draw(self):
        ts = pygame.Surface((1500,912))
        ts.set_colorkey((0,0,0))
        ts.blit(self.topimage,(self.x-self.w/2,self.y-self.h))
        ts.blit(self.bottomimage,(self.x-self.w/2,self.y+self.gap))
        screen.blit(ts,(0,0))

    def collide(self,bir):
        if pygame.Rect(self.x-self.w/2,0,self.w,self.y).colliderect(pygame.Rect(bir.x-bir.w/2,bir.y-bir.h/2,bir.w,bir.h)) or pygame.Rect(self.x-self.w/2,self.y+self.gap,self.w,1000).colliderect(pygame.Rect(bir.x-bir.w/2,bir.y-bir.h/2,bir.w,bir.h)):
            return True
        return False


class MAIN:
    def __init__(self):
        self.trainingmode = False
        self.aiplayer = True
        if self.trainingmode:
            self.batchsize = 100
            self.generation = 0
            self.gentimer = time.time()
            self.newnets = [ai.AI(170+160*a,20,140,200,7,5,1,1,'flappy bird ai') for a in range(self.batchsize)]
        elif self.aiplayer:
            self.newnets = [ai.AI(170,20,140,200,7,5,1,1,'flappy bird ai')]
        self.maxscore = 1
        
        self.birbx = 100
        self.wallvelocity = 10
        
        self.background = pygame.image.load('background.png')

        self.done = False
        self.clock = pygame.time.Clock()
        self.score = 0
        self.gengame()

        self.drawscreen = True
        self.drawnets = False

        
    def main(self):
        while not self.done:
            self.spacedown = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                    if event.key == pygame.K_SPACE:
                        self.spacedown = True
                    if event.key == pygame.K_F3:
                        if not self.drawnets: self.drawnets = True
                        else: self.drawnets = False
                    if event.key == pygame.K_F5:
                        if not self.drawscreen: self.drawscreen = True
                        else: self.drawscreen = False
                    
                        
            self.gametick()
            self.draw()
            self.clock.tick(600)
    def draw(self):
        screen.blit(self.background,(0,0))
        if self.drawscreen:
            for a in self.walls:
                a.draw()
            for b in self.birbs:
                if not b.dead: b.draw(self.wallvelocity/self.difficulty**0.5)
            birdsalive = 0
            for b in self.birbs:
                if not b.dead:
                    birdsalive+=1
                    if b.aicontrolled: b.net.setdisplay(90+160*birdsalive,20,140,200)
            if self.drawnets:
                for b in self.birbs:
                    if not b.dead and b.aicontrolled: b.net.displaynetwork(screen,[0,0,0,0])
            if self.trainingmode:
                write(10,110,'Alive:'+str(birdsalive),(200,200,200),60)
        write(10,60-50*(int(self.trainingmode)*-1+1),'Score:'+str(self.score),(200,200,200),60)
        if self.trainingmode: write(10,10,'Gen:'+str(self.generation),(200,200,200),60)
        write(10,960,str(int(self.clock.get_fps())),(200,200,200),30)
        pygame.display.flip()
        
    def gametick(self):
        if self.difficulty>0.5: self.difficulty*=0.9999
        self.walltimer-=1
        if self.walltimer<0:
            self.walltimer = random.gauss(80,5)*(self.difficulty**2)
            self.walls.append(wall(self.difficulty**0.5))
            
        for a in self.walls:
            a.move(self.wallvelocity/self.difficulty)
            for b in self.birbs:
                if not b.dead and a.collide(b):
                    b.die()
            if a.x<self.birbx and not a.givenscore:
                a.givenscore = True
                self.score+=1
                for b in self.birbs:
                    if not b.dead: b.blackbox[1]+=1
        if self.walls[0].x<-100:
            del self.walls[0]

        for b in self.birbs:
            if not b.dead:
                b.control(self.spacedown,self.walls,self.difficulty)
                b.move()

            
        for b in self.birbs:
            if (b.y>912-b.h/2 or b.y<0) and not b.dead:
                b.die()
        if sum([int(b.dead) for b in self.birbs]) == len(self.birbs):
            if self.trainingmode: self.darwanism()
            self.gengame()
                
        
        self.clock.tick(60)
    def gengame(self):
        if self.score>self.maxscore:
            self.maxscore = self.score
        if not self.trainingmode:
            if not self.aiplayer: print('your score was '+str(self.score))
            self.birbs = [BIRB(self.birbx,400,False,0)]
            if self.aiplayer:
                self.birbs.insert(0,BIRB(self.birbx,400,True,1))
                self.birbs[0].net = copy.deepcopy(self.newnets[0])
                
        else:
            self.birbs = [BIRB(self.birbx,400,True,a) for a in range(self.batchsize)]
            for i,b in enumerate(self.birbs):
                b.net = copy.deepcopy(self.newnets[i])
        self.walls = [wall(1)]
        self.walltimer = 100
        self.score = 0
        self.difficulty = 1
        
    def darwanism(self):
        print('-------------------------------- generation number ',self.generation,'--------------------------------')
        print('time lasted:',time.time()-self.gentimer)
        self.gentimer = time.time()
        gdata = []
        for a in self.birbs:
            gdata.append([a.net,a.blackbox[0],a.blackbox[1]])
        gdata = sorted(gdata, key=lambda x: x[1],reverse=True)
        print('best ais score:',gdata[0][2])
        

        a = 0
        self.newnets = [copy.deepcopy(gdata[a][0])]
        newgen = []
        while len(self.newnets)<self.batchsize:
            tnet = copy.deepcopy(gdata[a][0])
            children = tnet.evolvesingle(16,0.1*(1/self.maxscore))
            for b in range(len(children)):
                self.newnets.append(children[b])
            newgen.append((gdata[a][1]))
            a+=1
        print('ais that reproduced:',newgen)
        self.generation+=1
        self.birbs = [BIRB(self.birbx,400,True,a) for a in range(self.batchsize)]
        for i,b in enumerate(self.birbs):
            b.net = copy.deepcopy(self.newnets[i])










main = MAIN()
main.main()
pygame.quit()









