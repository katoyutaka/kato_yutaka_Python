import cv2
import matplotlib.pyplot as plt

pic_path = "./sample1.jpg"

cascade_path = "./haarcascades/haarcascade_frontalface_alt.xml"

omen_path = "../img/omen.jpeg"

#画像読み込み
image = cv2.imread(pic_path)
omen_image = cv2.imread(omen_path)

#読み込んだbgrをrgbに変換
rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
f= cv2.cvtColor(omen_image,cv2.COLOR_BGR2RGB)


#透明度αがないのでそれに変換
rgb_omen_image = cv2.cvtColor(f, cv2.COLOR_RGB2RGBA)

plt.imshow(rgb_image)
plt.show()

plt.imshow(rgb_omen_image)
plt.show()



#画像をグレースケール化
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



#カスケード検出器作成
cascade = cv2.CascadeClassifier(cascade_path)

#顔検出処理
face = cascade.detectMultiScale(gray_image,minSize = (10,10))


#お面を上に上書きする
for(x,y,w,h) in face:
    #縦と横で長い方を採用
    length = w
    if length < h:
        length = h
        
   #お面のサイズがデカいので小さくする
    resize_omen_image = cv2.resize(rgb_omen_image,(length, length))
    #お面を貼る
    rgb_image[y:length + y, x:length + x] = rgb_image[y:length + y, x:length + x] * (1 -resize_omen_image[:, :, 3:] / 255) + resize_omen_image[:, :, :3] * (resize_omen_image[:, :, 3:] / 255)
    


plt.imshow(rgb_image)
plt.show()