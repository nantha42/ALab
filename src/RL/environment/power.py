import pygame as py 
import typing
import numpy as np 
from icecream import ic
np.random.seed(5)


ENABLE_DRAW = True 

class PowerGame:
    def __init__(self,gr=10,gc=10,vis=7,neural_image = False):
        py.init()
        self.font = py.font.SysFont("times",20)
        self.box_size = 20 
        self.grid = np.zeros((gr,gc))
        self.vissize = vis
        extra_width = 500
        if self.display_neural_image:
            self.w = 50 + gc*self.box_size + 50 + extra_width  
        else:
            self.w = 50 + gc*self.box_size + 50 
        self.h = 50 + gr*self.box_size + 50
        self.win = py.display.set_mode((self.w,self.h),py.DOUBLEBUF,32)
        self.win.set_alpha(128)
        self.agents = None 
        self.test = [0,0]
        self.clock = py.time.Clock()
        self.start_position= [50,50]
        self.timer = 0
        self.res = 0
        self.items = 0
        self.collected = 0
        self.processors = 0
        self.agent_pos = np.array([0,0])
        self.enable_draw = ENABLE_DRAW 
        self.visibility = None
        self.processors = []
        self.current_step = 0
        self.total_rewards = 0
        
        self.hidden_state = None
        self.hidden_state_surfaces = []
        self.neural_image_values = []
        self.neural_image = None  # stores the surface
        self.neural_layout = None
        self.neural_layout_size = None


        self.display_neural_image = neural_image 
        self.neural_weights = None
        self.neural_weight_surface = None
        self.weight_change = False
 
        self.FOV = 90

        #initializing agents
        self.RES = 9
        self.PROCESSOR = 8
        self.ITEM = 7

    def render_text(self,text,pos):
        text = self.font.render(text,True,(200,200,200))
        trect = text.get_rect()
        trect.topleft =  pos 
        self.win.blit(text,trect)

   
    def get_state(self) -> np.ndarray :
        x,y = self.agent_pos
        vis = np.ones((self.vissize,self.vissize))*-1
        v,h = self.grid.shape
        for i in range(-2,3):
            for j in range(-2,3):
                if(0 <= x+i < v and 0 <= y+j < h):
                    vis[i+2,j+2] = self.grid[x+i,y+j] 
        self.visibility = vis
        return self.visibility 


    def act(self,action:np.ndarray):
        left,up,right,down = action[:4]
        v,h = self.grid.shape
        collect = action[4]
        build_proc = action[5]
        cx,cy = self.agent_pos
        
        reward = 0
        if left > 0 and self.agent_pos[1]>0:
            self.agent_pos[1] -=1
        elif right > 0 and self.agent_pos[1]< h-1:
            self.agent_pos[1] += 1
        elif up > 0 and self.agent_pos[0] >0:
            self.agent_pos[0] -=1 
        elif down>0 and self.agent_pos[0] < v-1:
            self.agent_pos[0] +=1
        elif collect > 0 and (self.grid[cx][cy] == self.RES or self.grid[cx][cy] == self.ITEM ):
            if self.grid[cx][cy] == self.RES:
                self.res -= 1
                reward = 1
                self.collected += 1
            elif self.grid[cx][cy] == self.ITEM:
                reward = 5
                self.items += 1
            self.grid[cx][cy] = 0
            reward = 1

        elif build_proc > 0 and self.collected >= 7 and self.grid[cx][cy] == 0:
            self.grid[cx][cy] = self.PROCESSOR
            self.processors.append([cx,cy])
            self.collected  -= 7  # 7 resources = 1 processor
            self.reward = 2
        
        new_state = self.get_state()
        self.total_rewards += reward
        return new_state,reward
        

    def reset(self,hard=False):
        if hard:
            np.random.seed(5)
        v,h = self.grid.shape
        self.grid = np.zeros((v,h))
        self.agent_pos = np.array([0,0]) 
        self.res = 0
        self.total_rewards =0
        self.current_step = 0
        self.items = 0
        self.collected = 0
        self.processors = []

        
    def project_point(self,point) :
        x,y,z = point
        inf = 1e9
        if z > 0:
            x_ = -np.arctan(np.radians(self.FOV/2))*x/z + 250 
            y_ = -np.arctan(np.radians(self.FOV/2))*y/z + 250 
        else:
            x_ = -inf
            y_ = -inf
        return [x_,y_]

    def create_pack(self,arr):
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:int(s*s)].reshape((s,s))
        col_splits = np.random.randint(3,6)
        columns = []
        a_split = s/col_splits
        total = s
        starting = 0
        for i in range(col_splits-1):
            end = np.random.randint(a_split-20,a_split)
            columns.append([starting,starting+end])
            starting+= end 
        columns.append([starting,s])

        final_split = []
        row_splits = 5 
        for column in columns:
            starting = 0
            a_split = s/row_splits
            rows = []
            for i in range(row_splits-1):
                e = np.random.randint(a_split-30,a_split+10)
                rows.append([starting,starting+e,column[0],column[1]])
                starting += e
            rows.append([starting,s,column[0],column[1]])
            final_split.append(rows)
        surface_size = [[s,s],[row_splits,col_splits]]
        return final_split,surface_size
    
    def weight_from_pack(self):
        arr = self.neural_weights
        l = arr.shape[0]
        s = int(np.sqrt(l))
        arr = arr[:(s*s)].reshape((s,s))
 
        out = []
        for col in self.neural_layout:
            rows = []
            for r in col:
                rows.append(arr[r[0]:r[1],r[2]:r[3]])
            out.append(rows)
        return out 
                
    def calculate_color(self,av,maxi,mini,poslimit,neglimit):
        cg = 0
        cr = 0
  
        if av < 0:
            cr = int( ((av)/mini)*neglimit)
            cg = 0
            # cg = poslimit- (cr)
        else:
            cg = int((av/maxi)*poslimit)
            # cr = neglimit - cg 
            cr = 0
        return cr,cg
    

    def draw_neural_image(self):
        self.neural_image = py.Surface( (500,self.h),py.SRCALPHA).convert_alpha()
        self.neural_image.fill((0,0,0))
        draw_dis = 10
        assert type(self.neural_image_values) == type(np.array([])), "neural_image_values should be in numpy array"
        if len(self.neural_image_values) < 1:
            return
        points = []
        varr = self.neural_image_values
        display_type = "SQUARE"
        if display_type == "CUBE":
            varr = self.neural_image_values
            varr = varr.reshape(-1) 
            l = int(np.cbrt(varr.shape[0]))
            varr = varr[:l**3].reshape((l,l,l))
            maxi = np.max(varr)
            mini = np.min(varr)
            rmax = 180
            gmax = 245
            if abs(maxi) > abs(mini):
                poslimit = 0 
                neglimit = int(rmax*( abs(mini)/ abs(maxi)))
            else:
                neglimit = rmax 
                poslimit = int(gmax*( abs(maxi)/ abs(mini)  ))
 
            for i in range(varr.shape[0]):
                for j in range(varr.shape[1]):
                    for k in range(varr.shape[2]):
                        x_,y_ = self.project_point(np.array([k/10-(varr.shape[2]//2)/10,j/10-(varr.shape[1]//2)/10,0.01+i/10]))
                        av =varr[i,j,k]
                        cr,cg = self.calculate_color(av,maxi,mini)
                        colorvalue = (abs(cr),0,abs(cg),)
                        py.draw.circle(self.neural_image,colorvalue,[x_,y_],1,1)
        else:
            varr = varr.reshape(-1)
            climit = int(np.sqrt(varr.shape[0]))
            sz = 10
            startx = 500/2-(climit/2)*sz
            starty = 500/4-(climit/4)*sz
            maxi = np.max(varr)
            mini = np.min(varr)
            rmax = 180
            gmax = 245
            if abs(maxi) > abs(mini):
                poslimit = 0 
                neglimit = int(rmax*( abs(mini)/ abs(maxi)))
            else:
                neglimit = rmax 
                poslimit = int(gmax*( abs(maxi)/ abs(mini)  ))

            i = 0
            r = 0
            c = 0
            while i < varr.shape[0]:
                av = varr[i]
                cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                py.draw.rect(self.neural_image,colorvalue,(startx+c*sz,starty+r*sz,sz,sz))
                c+=1 
                if c > climit:
                    c=0
                    r+=1
                i+=1

        if self.neural_weights is not None and self.weight_change:
            self.weight_change = False
            if self.neural_layout == None:
                self.neural_layout,self.neural_layout_size = self.create_pack(self.neural_weights)

            
            pix_size , gaps = self.neural_layout_size 
            gap_size = 1
            pix_size[0] += gaps[0] * gap_size
            pix_size[1] += gaps[1] * gap_size

            self.neural_weight_surface = py.Surface(pix_size)
            # self.neural_weight_surface.fill((250,0,0))

            weights = self.weight_from_pack()
            startx = 0
            for col in weights:
                starty = 0
                for weight in col:
                    r,c = weight.shape
                    maxi = np.max(weight)
                    mini = np.min(weight)
                    sz = 1.3

                    if abs(maxi) > abs(mini):
                       poslimit = 180 
                       neglimit = int(255.0*( abs(mini)/ abs(maxi)))
                    else:
                        neglimit = 255 
                        poslimit = int(255.0*( abs(maxi)/ abs(mini)  ))
                    
                    for i in range(r):
                        for j in range(c):
                            av = weight[i][j]
                            cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                            colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                            py.draw.rect(self.neural_weight_surface,colorvalue,(startx+j*sz,starty+i*sz,sz,sz))
                    starty += r*sz + gap_size 
                startx += c*sz + gap_size 
        if type(self.hidden_state) != type(None):
            maxi = np.max(self.hidden_state)
            mini = np.min(self.hidden_state)
            r,c = self.hidden_state.shape
            sz = 2 
            state_surface= py.Surface((c*sz,r*sz))
            print(state_surface.get_width(),state_surface.get_height())
            if abs(maxi) > abs(mini):
               poslimit = 180 
               neglimit = int(255.0*( abs(mini)/ abs(maxi)))
            else:
                neglimit = 255 
                poslimit = int(255.0*( abs(maxi)/ abs(mini)  ))

            for i in range(r):
                for j in range(c):
                    av = self.hidden_state[i][j]
                    cr,cg = self.calculate_color(av,maxi,mini,poslimit,neglimit)
                    colorvalue = (abs(cr),abs(cg),max(abs(cr),abs(cg)))
                    py.draw.rect(state_surface,colorvalue,(j*sz,i*sz,sz,sz))


            self.hidden_state_surfaces.append(state_surface)
            if len(self.hidden_state_surfaces) > 500:
                self.hidden_state_surfaces = self.hidden_state_surfaces[1:]

            for i in range(len(self.hidden_state_surfaces)):
                self.neural_image.blit(self.hidden_state_surfaces[i],
                                        (0,500*sz - i*r*sz+50))

        self.neural_image.blit(self.neural_weight_surface,
                                (self.neural_image.get_width()/2-self.neural_weight_surface.get_width()/2,self.h-self.h/2),special_flags = py.BLEND_RGB_ADD)
        self.win.blit(self.neural_image,(self.w-500,50),special_flags = py.BLEND_RGB_ADD)


    def draw_box(self,i,j,color):
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        # print(sx,sy)
        # print((sx+j*bs,sy+i*bs,bs,bs))
        surf = py.Surface((bs,bs),py.SRCALPHA)
        surf = surf.convert_alpha()
        py.draw.rect(surf,color,surf.get_rect())
        self.win.blit(surf,(sx+j*bs,sy+i*bs,bs,bs),special_flags=py.BLEND_RGBA_ADD)
#        py.draw.rect(self.win,py.Color(color),(sx+j*bs,sy+i*bs,bs,bs)) 


    def draw_grid(self):
        #horizontal line
        v,h = self.grid.shape
        r = 1
        bs = self.box_size#box size
        sx,sy = self.start_position 
        color = (100,0,0)

        for i in range(v+1):
            py.draw.line(self.win,color,(sx,sy+i*bs),(sx+h*bs,sy+i*bs))
            
        for i in range(h+1):
            py.draw.line(self.win,color,(sx+i*bs,sy),(sx+i*bs,sy+v*bs))


    def draw_visibility(self):
        if self.visibility is not None:
            vv,vh = self.visibility.shape
            ax,ay = self.agent_pos
            lx,ly = self.grid.shape
            for i in range(vv):
                for j in range(vh):
                    if 0<= i+ax-vv//2 < lx and 0<= j+ay-vh//2 < ly: 
                        self.draw_box(i+ax-vv//2,j+ay-vh//2,[50,50,50])

 
    def draw_items(self):
        v,h = self.grid.shape
        #v=20 h = 30
        for i in range(v):
            for j in range(h):
                if self.grid[i][j] == self.RES:
                    self.draw_box(i,j,(0,255,0))
                elif self.grid[i][j] == self.PROCESSOR:
                    self.draw_box(i,j,(255,0,0))
                elif self.grid[i][j] == self.ITEM:
                    self.draw_box(i,j,(0,255,255))


    def draw(self):
        self.win.fill((0,0,0))
        self.draw_grid()
        self.draw_items()
        self.render_text(f"Collected :{self.collected:3}",(0,0))
        self.render_text(f"Items     :{self.items:3}",(0,20))
        self.render_text(f"Steps     :{self.current_step:4}",(250,0))
        self.render_text(f"Rewards :{self.total_rewards:3}",(250,20))
        # self.agent_pos[1] = ((self.agent_pos[1]+1)%30)
        # if self.agent_pos[1] == 0:
        #     self.agent_pos[0] = ((self.agent_pos[0]+1)%20)
        # # print(self.agent_pos)
        # print(self.get_state())
        self.get_state()
        self.draw_box(self.agent_pos[0],self.agent_pos[1],(0,0,200))
        self.draw_visibility()
        if self.display_neural_image:
            self.win.blit(self.neural_image, (self.w-500,0))
        py.display.update()
    
    def spawn_resources(self):
        #initializing foods
        self.timer +=1
        v,h = self.grid.shape
        if self.timer%10 == 0 and self.res < 10:
            while True: 
                r = np.random.randint(0,v)
                c = np.random.randint(0,h)
                if self.grid[r][c] == 0:
                    self.grid[r][c] = self.RES
                    self.res +=1
                    break
        if self.timer%10 == 0:
            for processor in self.processors:
                px,py = processor
                if self.collected > 3:
                    if px -1 > -1 and self.grid[px-1][py] == 0:
                        self.grid[px-1][py] = self.ITEM
                        self.collected -= 3
                    elif px+1 < v and self.grid[px+1][py] == 0:
                        self.grid[px+1][py] = self.ITEM
                        self.collected -= 3
                    elif py -1 > -1 and self.grid[px][py-1] == 0:
                        self.grid[px][py-1] = self.ITEM
                        self.collected -= 3
                    elif py+1 < h and self.grid[px][py+1] == 0:
                        self.grid[px][py+1] = self.ITEM
                        self.collected -= 3
                else:
                    break
        pass

    def event_handler(self):
        for event in py.event.get():
            if event.type == py.QUIT:
                py.quit()
                exit()
    
    def update(self):
        self.spawn_resources()
        pass

    def step(self,speed=0):
        self.event_handler()
        self.update()
        self.current_step += 1
        if self.enable_draw:
            if self.display_neural_image:
                self.draw_neural_image()
            self.draw()
            self.clock.tick(speed)

if __name__ == '__main__':
    r = PowerGame() 
    # print(r.get_state())
    while True:
        r.step()

