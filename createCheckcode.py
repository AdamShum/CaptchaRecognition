
from PIL import Image, ImageFont, ImageDraw
from numpy import random

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number, captcha_size=4):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return ''.join(captcha_text)
 



class captcha(object):
    def __init__(self, text='captcha',
        width=72,height=27,fontsize=25,border=5):
        self.font = ImageFont.truetype('./font.ttf', size=fontsize)
        self.text = text
        self.width = width
        self.height = height
        self.fontsize = fontsize
        self.border = border
        self.image = None
    def get_font_size(self):
        return self.font.getsize(self.text)
    def getsize(self):
        if self.width == 0:
            self.width = self.get_font_size()[0] + 2*self.border
        if self.height == 0:
            self.height = self.get_font_size()[1] + 2*self.border
        return (self.width, self.height)
    def getimage(self):
        img = Image.new(mode="RGB", size=self.getsize(), color=(255,255,255))
        draw = ImageDraw.Draw(img)
        draw.text((self.border,self.border), self.text, font=self.font, fill="#000")
        self.image = img
        return self.image
    def save(self,filepath='./test.png'):
        if not self.image:
            self.getimage()
        self.image.save(filepath)


def test():
    for i in range(60000):
        text=random_captcha_text()
        a = captcha(text)
        
        
        a.save('./img/'+str(i)+'_'+text+'.png')

if __name__ == '__main__':
    test()

