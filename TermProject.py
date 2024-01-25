#----------------라이브러리, 변수 선언---------------
import numpy as np
import cv2
import math

#드래그 발생시 이전의 좌표값 저장
prevX = 0   
prevY = 0

#마우스 이벤트가 일어나는 실시간 좌표
nowX = 0    
nowY = 0

isDragging = False                          #드래그가 일어나는 중인지 확인할 변수
selectConvolution = 0                       #어떤 회선을 수행할 것인지 지정할 변수
isConvolutionPoint = np.zeros((1920,1080))  #회선이 수행된 곳을 담을 변수

#글씨가 있는 부분은 회선이 되지 않도록 영역 지정 
# 0 : 회선이 진행되지 않았음  /  1 : 회선이 일어남 /  2 : 회선을 수행하지 않을 곳  
for i in range(0,80):
    for j in range(0,1100):
        isConvolutionPoint[j,i] = 2
#--------------------------------------------





#----------------회선 마스크 생성---------------
blurMask = [ 1/9 for _ in range(9)]
blurMask = np.array(blurMask, np.float32).reshape(3,3)

sharpeningMask = [-1, -1, -1, -1, 9, -1, -1, -1, -1]
sharpeningMask = np.array(sharpeningMask, np.float32).reshape(3,3)

prewittMask1 = [-1,0,1,-1,0,1,-1,0,1]
prewittMask1 = np.array(prewittMask1, np.float32).reshape(3,3)
prewittMask2 = [-1,-1,-1,0,0,0,1,1,1]
prewittMask2 = np.array(prewittMask2, np.float32).reshape(3,3)

sobelMask1 = [-1,0,1,-2,0,2,-1,0,1]
sobelMask1 = np.array(sobelMask1, np.float32).reshape(3,3)
sobelMask2 = [-1,-2,-1,0,0,0,1,2,1]
sobelMask2 = np.array(sobelMask2, np.float32).reshape(3,3)

laplacianMask1 = [-1,-1,-1,-1,8,-1,-1,-1,-1]
laplacianMask1 = np.array(laplacianMask1, np.float32).reshape(3,3)
laplacianMask2 = [1,1,1,1,8,1,1,1,1]
laplacianMask2 = np.array(laplacianMask2, np.float32).reshape(3,3)
#--------------------------------------------






#----------------함수 정의--------------------------

#좌표 x,y에서 반지름 30인 원의 좌표들을 반환하는 함수
def getConvolutionArea(x, y):

    #회선 영역의 최대 최소값이 해상도를 벗어나지 않게 함
    x_min = max(0, x - 30)
    x_max = min(1920, x + 30 + 1)
    y_min = max(0, y - 30)
    y_max = min(1080, y + 30 + 1)

    #원 영역의 좌표를 담을 배열 생성 ret
    ret = []

    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            if (i - x) ** 2 + (j - y) ** 2 <= 900:      #원의 정의에 부합하는 점이고
                if isConvolutionPoint[i,j] == 0:        #회선이 수행되지 않은 점이면     
                    ret.append((i,j))                   #해당 좌표는 회선할 대상이므로 저장

    ret = np.array(ret).astype(int)                     #numpy 배열로 변환하고, 정수 좌표로 변경 후 반환
    return ret





#블러링, 샤프닝 회선 수행함수 
def filter1(image, mask, x, y):
    dst = image.copy()                              #원본이미지 복사 (블러링 지점말고는 기존 이미지를 사용하므로)
    convolutionArea = getConvolutionArea(x, y)      #원의 좌표를 받아오기

    #회선을 수행할 좌표에 회선 수행 
    for point in convolutionArea:
        j,i = point
        roi = originalInputImg[i - 1: i + 2, j - 1:j + 2].astype('float32')    # 원본 영상에서 3x3 pixel을 추출

        #각 채널에 마스크를 적용함 B,G,R 채널에 각각 수행함
        for channel in range(3):
            tmp = roi[:, :, channel] * mask
            result = np.sum(tmp)
            dst[i, j, channel] = np.clip(result, 0, 255)     # 0~255로 클리핑하여 색상 반전이 일어나지 않게함       

        #회선 수행한 곳은 1로 적용하여 회선을 여러번 수행하지 않게 함 
        isConvolutionPoint[j,i] = 1
    return dst

#프리윗, 소벨, 라플라시안 회선 수행 함수 
def filter2(image,mask1,mask2,x,y):
    dst = image.copy()
    convolutionArea = getConvolutionArea(x, y)

    for point in convolutionArea:
        j,i = point
        roi = originalInputImg[i - 1: i + 2, j - 1:j + 2].astype('float32')

        for channel in range(3):    
            res1 = np.sum(roi[:, :, channel] * mask1)       #magnitude 를 위해 두 마스크의 값을 각각 적용
            res2 = np.sum(roi[:, :, channel] * mask2)
            magnitude = np.sqrt(res1 ** 2 + res2 ** 2)      #magnitude 계산
            dst[i, j, channel] = np.clip(magnitude, 0, 255) 

        isConvolutionPoint[j,i] = 1
    return dst



#마우스 이벤트가 일어날 때 두 지점을 잇는 직선상의 점들을 추출 
def getLinePoints(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1       #반복마다 원소값 증감을 결정
    sy = -1 if y0 > y1 else 1
    err = dx - dy                   #반드시 45도 직선이 아니므로, 오차계산 변수

    points = []
    intervalCount = 0

    #특징점만을 추출하기 위해 30픽셀마다 한번씩 저장
    interval = 30                   
    while x0 != x1 or y0 != y1:
        if intervalCount == interval:
            points.append((x0, y0))
            intervalCount = 0

        #좌표는 정수형이므로 오차를 연산하여 직선상에 위치하도록 조정 
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        
        intervalCount += 1

    points.append((x1, y1))  # 종료점도 포함시킨다.
    return points




#마우스 콜백함수
def onMouse(event, x,y, flags, param):
    global isDragging,inputImg,prevX,prevY,nowX,nowY        #함수 수행에 필요한 전역변수 가져오기 

    #마우스 드래그가 일어나지 않고, 이동만 일어나고 있는 경우
    if event == cv2.EVENT_MOUSEMOVE and isDragging == False:
        previousImg = inputImg.copy()                               #원본 이미지 복사    
        cv2.circle(previousImg, (x,y), 30, (0,0,255), -1)           #원을 복사된 이미지에 그리기
        dst = cv2.addWeighted(inputImg , 0.5 , previousImg, 0.5,0)  #적절히 합성하여 반투명하게 나타내기
        cv2.imshow(title, dst)                                      #합성된 이미지 출력 

    #마우스를 누른 경우, 현재 클릭된 좌표값을 저장한다. 
    #그리고 드래깅이 일어나는 중이라 체크한다.
    if event == cv2.EVENT_LBUTTONDOWN: 
        isDragging = True
        prevX = x
        prevY = y
        nowX = x
        nowY = y

    # 누른 상태에서 이동 (드래그) 발생하는 경우, 좌표값을 최신화하고, 직선 상의 점을 가져온다.
    elif event == cv2.EVENT_MOUSEMOVE and isDragging == True:
        prevX = nowX
        prevY = nowY
        nowX = x
        nowY = y
        lines = getLinePoints(prevX, prevY, nowX, nowY)     #해당 직선의 점들을 가져오기 

        #각 회선 선택 변수에 맞게 회선 호출을 한다. 
        if selectConvolution == 1:
            for line in lines:
                x,y = line
                inputImg = filter1(inputImg, blurMask, x,y)
            cv2.imshow(title,inputImg)

        elif selectConvolution == 2:
            for line in lines:
                x,y = line
                inputImg = filter1(inputImg, sharpeningMask, x,y)
            cv2.imshow(title,inputImg)

        elif selectConvolution == 3:
            for line in lines:
                x,y = line
                inputImg = filter2(inputImg, prewittMask1,prewittMask2, x,y)
            cv2.imshow(title,inputImg)

        elif selectConvolution == 4:
            for line in lines:
                x,y = line
                inputImg = filter2(inputImg, sobelMask1,sobelMask2, x,y)
            cv2.imshow(title,inputImg)

        elif selectConvolution == 5:
            for line in lines:
                x,y = line
                inputImg = filter2(inputImg, laplacianMask1,laplacianMask2, x,y)
            cv2.imshow(title,inputImg)
    
    #마우스 클릭을 종료하는 경우 드래그 종료되었다고 저장 
    elif event == cv2.EVENT_LBUTTONUP:
        isDragging = False
#--------------------------------------------



#----------------이미지 불러오기, 이벤트처리 등록---------------
title = 'input.jpg'                                     #제목 설정 
inputImg = cv2.imread("input.jpg", cv2.IMREAD_COLOR)    #컬러영상으로 가져오기
inputImg = cv2.resize(inputImg, (1920, 1080))           #해상도에 맞게 조정
originalInputImg = inputImg.copy()                      #회선 시 참조할 원본 영상값
cv2.imshow(title,inputImg)

#회선 선택 글씨 나타내기 
cv2.putText(inputImg, "(1) Blurring (2) Shapening (3) Prewwit (4) Sobel (5) Laplacian", (0,50) , cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255) , 3)

#콜백함수 등록하기. 
cv2.setMouseCallback(title, onMouse)
while True:
    key = cv2.waitKey(100)

    #이미지 저장, 회선이 수행된 inputImg 는 바로 저장
    #for loop를 돌려 회선이 수행된 좌표는 화소값을 0으로 처리하여 저장 
    if(key == ord('q')):
        cv2.imwrite("output1.jpg", inputImg)

        #회선 수행된 곳은 검은색으로 화소값을 저장 
        for i in range(1920):
            for j in range(1080):
                if(isConvolutionPoint[i,j] == 1):   #회선을 수행한 곳이면 
                    inputImg[j,i] = (0,0,0)         #화소값을 0으로 만들기
        cv2.imwrite("output2.jpg", inputImg)

        cv2.destroyAllWindows()

    # 키보드 입력 값이 1~5 인 경우, 드래그가 일어나지 않을 때 회선 선택 번호 변경 
    # 현재 회선의 종류를 출력해줌 (생략 가능)
    elif(key == ord('1') and isDragging == False):
        print("Blurring")
        selectConvolution = 1
    elif(key == ord('2') and isDragging == False):
        print("Sharpnning")
        selectConvolution = 2
    elif(key == ord('3') and isDragging == False):
        print("Prewitt")
        selectConvolution = 3
    elif(key == ord('4') and isDragging == False):
        print("Sobel")
        selectConvolution = 4
    elif(key == ord('5') and isDragging == False):
        print("Laplacian")
        selectConvolution = 5