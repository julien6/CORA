"""Contextual illustrations of actual states; scenery does not add simulator physics."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

COLORS=['#e56b58','#2aaf8a','#5689dc']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(size=16):return ImageFont.truetype(FONT,size)
def text(d,xy,s,fill='#24364b',size=16):d.text(xy,str(s),fill=fill,font=font(size))
def tree(d,x,y,r=18):
    d.line((x,y,x,y+r),fill='#785b3f',width=5)
    d.ellipse((x-r,y-r,x+r,y+r*.4),fill='#37845e',outline='#256443',width=2)
def agent(d,x,y,i,domain,failed=False):
    c=COLORS[i];d.ellipse((x-23,y+10,x+23,y+22),fill='#bbc5c6')
    if domain=='warehouse':
        d.rounded_rectangle((x-19,y-14,x+19,y+16),radius=6,fill=c,outline='white',width=2)
        for dx in (-18,18):d.rectangle((x+dx-3,y-9,x+dx+3,y+10),fill='#233444')
        d.rectangle((x-10,y-8,x+10,y+4),fill='#e5ddd0')
    else:
        d.ellipse((x-8,y-27,x+8,y-11),fill='#f1c4a3',outline='#53493d')
        d.rounded_rectangle((x-14,y-12,x+14,y+12),radius=5,fill=c,outline='white',width=2)
        d.line((x-7,y+12,x-9,y+23),fill='#25384a',width=5);d.line((x+7,y+12,x+9,y+23),fill='#25384a',width=5)
        if domain in ('hospital','rescue'):
            d.line((x-6,y,x+6,y),fill='white',width=3);d.line((x,y-6,x,y+6),fill='white',width=3)
        if domain=='wildfire':d.arc((x-10,y-30,x+10,y-12),180,360,fill='#f9be42',width=7)
    text(d,(x-7,y-6),i,fill='white',size=13)
    if failed:
        d.ellipse((x-29,y-33,x+29,y+27),outline='#be2635',width=3)
        d.line((x-22,y-25,x+22,y+20),fill='#be2635',width=5)


def application_frames(trace,env,assignments,title):
    states=trace['states'];frames=[];domain=env.domain
    names={'warehouse':['Reception','Quality control','Dispatch'],'rescue':['Locate','Medical aid','Evacuation'],'wildfire':['Sector A','Sector B','Sector C'],'hospital':['Triage','Treatment','Discharge']}[domain]
    def point(p):return (int(82+p[0]*107),int(530-p[1]*99))
    sites=[point(p) for p in np.asarray(env.sites)]
    for t,pos in enumerate(np.asarray(states.positions)):
        im=Image.new('RGB',(1000,650),'#eef2f6');d=ImageDraw.Draw(im)
        d.rectangle((0,0,1000,76),fill='#172b40');text(d,(24,12),domain.upper()+'  |  '+title.replace('_',' '),'white',23)
        text(d,(24,46),f'Step {t:02d} / {len(states.positions)-1}  •  actual simulator state','#b8d1e4',15)
        d.rounded_rectangle((22,92,608,587),radius=12,fill={'warehouse':'#e3e7e9','hospital':'#e7f1f3','rescue':'#d5e3c9','wildfire':'#dfdcc0'}[domain],outline='#a9b8bd',width=2)
        # The grid is the actual movement lattice, scenery is decorative.
        for x in range(5):
            a=point((x,0));b=point((x,4));d.line((*a,*b),fill='#c4d0cc',width=1)
        for y in range(5):d.line((*point((0,y)),*point((4,y))),fill='#c4d0cc',width=1)
        if domain=='warehouse':
            for x,y in [(105,160),(265,160),(470,160),(95,320)]:
                d.rounded_rectangle((x-28,y-22,x+28,y+22),radius=3,fill='#647b86')
                for dx in (-17,4):d.rectangle((x+dx,y-14,x+dx+14,y+13),fill='#c49b64',outline='#8b653b')
            d.line((50,564,567,564),fill='#d1a644',width=6);text(d,(54,566),'ROBOT TRANSPORT AISLE',size=11)
        elif domain=='hospital':
            for x,y in [(92,165),(257,165),(495,265)]:
                d.rectangle((x-27,y-12,x+27,y+18),fill='white',outline='#92a9b4',width=3)
                d.rectangle((x-22,y-8,x-4,y+12),fill='#b6dbe6');d.line((x+18,y+18,x+18,y+29),fill='#92a9b4',width=3)
            d.rectangle((38,104,598,129),fill='#78acb4');text(d,(45,106),'SIMULATED CARE UNIT',fill='white',size=13)
        else:
            for x,y in [(75,135),(290,180),(530,130),(70,280),(560,390),(268,542)]:tree(d,x,y)
            if domain=='rescue':
                d.polygon([(65,193),(102,167),(139,193),(139,230),(65,230)],fill='#a3917e',outline='#6f6153')
                d.line((70,200,100,211,91,230),fill='#5c5147',width=3)
                d.rectangle((510,485,572,525),fill='white',outline='#758c93',width=2);text(d,(533,489),'+',fill='#cc4949',size=26)
        for i,(x,y) in enumerate(sites):
            d.rounded_rectangle((x-45,y-33,x+45,y+32),radius=8,fill='#fafbfc',outline='#8fa0ad',width=2)
            if domain=='wildfire':
                tree(d,x,y,21)
                if states.queue[t,i]>0:
                    for dx in (-18,3,18):d.polygon([(x+dx-10,y+15),(x+dx,y-22),(x+dx+13,y+15)],fill='#e78237',outline='#bf542e')
            elif domain=='hospital':
                d.rectangle((x-22,y-12,x+22,y+14),fill='#bbdbe2',outline='#6a99a8');text(d,(x-9,y-16),'+',fill='white',size=25)
            elif domain=='rescue':
                d.polygon([(x-32,y+12),(x,y-24),(x+32,y+12)],fill='#ece3c4',outline='#ab9570');d.rectangle((x-8,y,x+8,y+19),fill='#83745e')
            else:
                for dx in (-22,0,22):d.rectangle((x+dx-8,y-10,x+dx+8,y+10),fill='#c49b64',outline='#8b653b')
            text(d,(x-52,y+36),names[i],size=12)
            text(d,(x-32,y-57),f'Queue: {float(states.queue[t,i]):.0f}',size=13)
        assigned=np.asarray(assignments[t] if np.ndim(assignments)==2 else assignments)
        for i,p in enumerate(pos):
            history=np.asarray(states.positions)[:t+1,i]
            if len(history)>1:d.line([point(v) for v in history],fill=COLORS[i],width=3)
            x,y=point(p);agent(d,x,y,i,domain,bool(states.failed[t,i]))
        d.rounded_rectangle((627,92,978,587),radius=12,fill='white')
        text(d,(645,111),'MISSIONS & AGENT STATUS',size=18)
        for i in range(3):
            y=151+i*91;d.ellipse((646,y,661,y+15),fill=COLORS[i]);text(d,(671,y-3),f'Agent {i}  '+('FAILED' if states.failed[t,i] else 'active'),size=17)
            text(d,(671,y+23),'Assigned: '+names[int(assigned[i])],size=14)
            text(d,(671,y+45),'Position: '+str(np.asarray(pos[i]).tolist()),fill='#687e8b',size=12)
        text(d,(645,433),'TASK PROGRESS',size=18)
        for i in range(3):text(d,(645,463+25*i),f'{names[i]}: {float(states.completed[t,i]):.0f} completed',size=14)
        text(d,(645,548),f'Resource remaining: {float(states.resource[t]):.0f}',size=15)
        frames.append(im)
    return frames


def kitchen_image(state):
    """Readable kitchen scenery driven exclusively by native OvercookedV2 state."""
    from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME
    grid=np.asarray(state.grid);h,w=grid.shape[:2];size=112
    im=Image.new('RGB',(w*size,h*size),'#eee7da');d=ImageDraw.Draw(im)
    ingredient_colors=['#edc744','#58a657','#986bbc','#53bed0']
    def items(value,x,y):
        n=int(value)>>2;colors=[];idx=0
        while n:
            colors += [ingredient_colors[idx%4]]*(n&3);n>>=2;idx+=1
        if int(value)&1:d.ellipse((x-20,y-16,x+20,y+16),fill='white',outline='#a9bcc2',width=2)
        for i,c in enumerate(colors):
            dx=(i%3-1)*12;dy=(i//3)*12
            d.ellipse((x+dx-8,y+dy-8,x+dx+8,y+dy+8),fill=c,outline='#86795b')
    for y in range(h):
        for x in range(w):
            obj,value,extra=map(int,grid[y,x]);cx=x*size+size//2;cy=y*size+size//2
            if obj==0:
                d.rectangle((x*size,y*size,(x+1)*size,(y+1)*size),fill='#f1ecdf' if (x+y)%2 else '#e7e1d4',outline='#cfc9be')
                continue
            d.rectangle((x*size,y*size,(x+1)*size-1,(y+1)*size-1),fill='#a88766',outline='#5d5146',width=2)
            d.rectangle((x*size+5,y*size+5,(x+1)*size-6,(y+1)*size-6),fill='#c8b194')
            if obj==5:
                d.rounded_rectangle((cx-41,cy-37,cx+41,cy+34),radius=5,fill='#45545e')
                d.ellipse((cx-32,cy-31,cx+32,cy+27),fill='#22313c',outline='#e0e6df',width=3);items(value,cx,cy)
                d.line((cx-39,cy,cx-49,cy),fill='#25313c',width=6);d.line((cx+39,cy,cx+49,cy),fill='#25313c',width=6)
                text(d,(cx-16,cy+33),'POT',size=10)
                if extra:d.rectangle((cx-32,cy-43,cx-32+64*(1-extra/POT_COOK_TIME),cy-38),fill='#4fa77d')
                if value&2:text(d,(cx-28,cy-52),'READY',fill='#246443',size=11)
            elif obj==9:
                for dx,dy in [(-15,7),(15,7),(0,-15)]:d.ellipse((cx+dx-20,cy+dy-12,cx+dx+20,cy+dy+12),fill='white',outline='#9ab0b7',width=2)
                text(d,(cx-22,cy+36),'PLATES',size=10)
            elif obj>=10:
                d.rectangle((cx-39,cy-35,cx+39,cy+29),fill='#9d704a',outline='#68452c',width=3)
                for dx,dy in [(-19,-16),(17,-15),(0,8)]:
                    d.ellipse((cx+dx-14,cy+dy-13,cx+dx+14,cy+dy+13),fill=ingredient_colors[(obj-10)%4],outline='#675f35')
                text(d,(cx-32,cy+35),'INGREDIENT '+str(obj-10),size=9)
            elif obj==4:
                d.rectangle((cx-41,cy-39,cx+41,cy+20),fill='#dee8e8',outline='#5c787b',width=3)
                d.line((cx-38,cy-6,cx+38,cy-6),fill='#607f87',width=4);text(d,(cx-34,cy+31),'DELIVERY',fill='#305764',size=11)
            elif obj in (6,7):
                d.rectangle((cx-42,cy-34,cx+42,cy+28),fill='#29473f',outline='#c5ad80',width=4);items(state.recipe,cx,cy)
                text(d,(cx-23,cy+35),'RECIPE',size=11)
            else:
                if value:items(value,cx,cy)
    for i,(x,y) in enumerate(zip(np.asarray(state.agents.pos.x),np.asarray(state.agents.pos.y))):
        cx=int(x)*size+56;cy=int(y)*size+61;c=['#df6658','#507fce'][i]
        d.ellipse((cx-25,cy+18,cx+25,cy+32),fill='#c1b8aa')
        d.rounded_rectangle((cx-23,cy-12,cx+23,cy+27),radius=8,fill=c,outline='#3f4850',width=2)
        d.rectangle((cx-15,cy+4,cx+15,cy+26),fill='white');text(d,(cx-5,cy+6),i,fill=c,size=14)
        d.ellipse((cx-14,cy-31,cx+14,cy-8),fill='#efc6a5',outline='#967964')
        d.rounded_rectangle((cx-23,cy-43,cx+23,cy-23),radius=8,fill='white',outline='#b3c4cb',width=2)
        for dx in (-16,0,16):d.ellipse((cx+dx-9,cy-51,cx+dx+9,cy-31),fill='white',outline='#b3c4cb')
        items(state.agents.inventory[i],cx+30,cy+11)
        direction=int(state.agents.dir[i]);dx,dy=[(0,-1),(0,1),(1,0),(-1,0)][direction]
        d.line((cx+dx*28,cy+dy*28,cx+dx*40,cy+dy*40),fill=c,width=4)
    return im
