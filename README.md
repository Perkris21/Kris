import cv2

backSub_mog = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture("airport.mp4") # Пример с видео (аэропорт)

min_count = 100
max_count = 0
counts = []

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('airport_result.avi',fourcc, 20.0, (640, 480))

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    
    imgNoBg =  backSub_mog.apply(frame)
    
    contours, hierarchy = cv2.findContours(imgNoBg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    frame_ct = cv2.drawContours(imgNoBg, contours, -1, (0, 255, 0), 2)   

    min_contour_area = 800  

    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    frame_out = frame.copy()

    if len(large_contours) > max_count:
        max_count = len(large_contours)

    if len(large_contours) < min_count and len(large_contours) > 1:
        min_count = len(large_contours)

    if len(large_contours) > 1:
        counts.append(len(large_contours))

 
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
    
    cv2.putText(frame, f"Peoples in frame: {len(large_contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    
    cv2.putText(frame, f"Mimimal count in frame: {min_count}", (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    

    cv2.putText(frame, f"Maximal count in frame: {max_count}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
  
    if len(counts) > 0:
        cv2.putText(frame, f"Average count in frame: {round(sum(counts) / len(counts), 2)}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Frame_final', frame_out)

    out.write(frame_out)
    
    cv2.imshow('Mask', imgNoBg)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()

out.release()

cv2.destroyAllWindows()
