import cv2

cap = cv2.VideoCapture(0)

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

while(True):
    ret, frame = cap.read()    # Read 결과와 frame

    if(ret) :
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환

        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite('new.jpg', frame)
            break

cap.release()
cv2.destroyAllWindows()