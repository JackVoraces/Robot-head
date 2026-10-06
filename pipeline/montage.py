import json, sys
from PIL import Image, ImageDraw, ImageFont
files=json.load(open(sys.argv[1])); out=sys.argv[2]; cols=int(sys.argv[3]) if len(sys.argv)>3 else 5
ims=[Image.open(f).convert('RGB') for f in files]; w,h=ims[0].size
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',max(14,w//18))
rows=(len(ims)+cols-1)//cols; sheet=Image.new('RGB',(cols*w,rows*h),(20,20,20))
for i,(im,f) in enumerate(zip(ims,files)):
    x,y=(i%cols)*w,(i//cols)*h; sheet.paste(im,(x,y))
    if len(sys.argv)<=4 or sys.argv[4]!='nolabel':
        name=f.rsplit('_',1)[-1][:-4]; ImageDraw.Draw(sheet).text((x+8,y+6),name,fill=(255,255,255),font=font)
sheet.save(out); print(out,sheet.size)
