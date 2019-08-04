import cv2
import numpy as np

digits=cv2.imread("digits.png",cv2.IMREAD_GRAYSCALE)
test_digits=cv2.imread("test_digits.png",cv2.IMREAD_GRAYSCALE)

rows= np.vsplit(digits,50)
cells=[]  #in opencv we need speed so we use np.array
for row in rows:
    row_cells=np.hsplit(row,50)
   # cv2.imshow("row_cells0",row_cells[0]) 
    for cell in row_cells:
        cell=cell.flatten()
        #algo cannot work with more arrays so we use flatten to get one of them
        cells.append(cell)
       # print(cell)
        #cv2.imshow("c",cell)
cells=np.array(cells,dtype=np.float32) 

   
k=np.arange(10) #here k is just a array of 0-9 digits
#print(k)
cells_labels=np.repeat(k,250) #now for labels we are repeating the 0 to 250 times


test_digits=np.vsplit(test_digits,50)  

test_cells=[]

for d in test_digits:
    d=d.flatten()
    test_cells.append(d)

test_cells=np.array(test_cells,dtype=np.float32)
#KNN
knn=cv2.ml.KNearest_create() 

knn.train(cells,cv2.ml.ROW_SAMPLE,cells_labels) 

ret,result,neighbours,dist=knn.findNearest(test_cells,k=1)

print (result)
#cv2.imshow("row0",cells[250]) 
#cv2.imshow("digits",digits) #for showing the image with window name
cv2.waitKey(0)                            #digits,name of image is digits(1)
cv2.destroyAllWindows()               #wait keys wait indefinitely if 0 ispassed otherwise
                                     #to the specified duration
                                     #destroy all other windows created