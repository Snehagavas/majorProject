from deepface import DeepFace
import tensorflow
import cv2
import matplotlib.pyplot as plt   
img1=cv2.imread(r'C:\Users\Siddhi\Desktop\happy\PrivateTest_218533.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()              
result=DeepFace.analyze(img1, actions=['emotion'])                                        
  print(result)