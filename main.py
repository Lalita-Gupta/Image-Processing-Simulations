# # Importing Packages
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import streamlit as st 
from streamlit_drawable_canvas import st_canvas

# sidebar
with st.sidebar:
    main_choice = st.selectbox("Image Processing Simulations", ["Select one", "Basics", "Blending or Pasting", "Blurring or Smoothening", "Edge Detection", "Feature Detection", "Watershed Algorithm", "Feature Matching", "Template Matching", "Image Inprinting"])

# Basics
if main_choice == "Select one":
    pass

# Basics
if main_choice == "Basics":

    # title
    st.title("Run Simulation for Image Processing Basics")

    # choice:
    choice = st.selectbox("Chose activity to perform", ["Select one", "Color Conversion", "Image Resizing", "Image Slicing"])

    #for 
    if choice == "Color Conversion":

        conversion = st.selectbox("Pick one", ["Convert to Gray", "Convert to BGR", "Convert to Binary"])

        img = cv2.imread("/Users/lg/Documents/VirtualLab_ImageProcessing/Image_Processing_Simulations/Image set/Basics/basics_image1.webp")
        trial = cv2.imread("/Users/lg/Documents/VirtualLab_ImageProcessing/Image_Processing_Simulations/Image set/Basics/basics_image1.webp")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption = "Original Image")
        st.write("Original Image dimensions:", img.shape)

        if conversion == "Convert to BGR":
            st.image(trial, caption = "Original Image")
            st.write("Original Image dimensions:", trial.shape)

        if conversion == "Convert to Gray":
            trial = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(trial, caption = "Original Image")
            st.write("Original Image dimensions:", trial.shape)
        
        if conversion == "Convert to Binary":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, trial = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            st.image(trial, caption = "Original Image")
            st.write("Original Image dimensions:", trial.shape)

    #for 
    if choice == "Image Resizing":

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            horizontal = st.slider("Horizontal Resizing", 1, 4000, value = 400)
            vertical = st.slider("Vertical Resizing", 1, 1000, value = 400)

        with col2:
            img = cv2.imread("Experiment1/image.webp")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption = "Original Image")
            st.write("Original Image dimensions:", img.shape)

            img = cv2.resize(img,(horizontal,vertical))
            st.image(img, caption = "Original Image")
            st.write("Original Image dimensions:", img.shape)

    
    if choice == "Image Slicing":

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            bl = st.slider("Bottom Left Corner", 1, 800, value = 100)
            br = st.slider("Bottom Right Corner", 1, 800, value = 100)

        with col2:
            img = cv2.imread("Experiment1/image.webp")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption = "Original Image")
            st.write("Original Image dimensions:", img.shape)

            cropped = img[0:bl, 0:br]
            st.image(cropped, caption = "Original Image")
            st.write("Original Image dimensions:", cropped.shape)

# SIMULATION 
if main_choice == "Blending or Pasting":

    # title
    st.title("Run Simulation for Blending and Pasting")


    # choice of shape:
    choice = st.selectbox("Blending or Pasting", ["Select one", "Blending", "Pasting"])
    #********************************************

    # For Blending
    if choice == "Blending":
        
        blend_choice = st.selectbox("Same or Different Dimension", ["Select one", "Blend Image of Same Size", "Blend Image of Different Size"])

        # Blending images of same size
        if blend_choice == "Blend Image of Same Size":

            #creating of columns
            col1, col2 = st.columns([1, 3])
                    
            with col1: 
                alpha_value = st.slider("Choose value of Alpha", 0.0, 1.0, value = 0.5)
                beta_value = st.slider("Choose value of Beta", 0.0, 1.0, value = 0.5)
                gamma_value = st.slider("Choose value of Gamma", 0, 100, value = 0)

            with col2:
                large_img = cv2.imread("Experiment3/images/image1.jpeg")
                large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)
                large_img = cv2.resize(large_img,(1024,1024))
                small_img = cv2.imread("Experiment3/images/image2.png")
                small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                st.image(large_img, caption = "First Image")
                st.write("First Image dimensions:", large_img.shape)
                st.image(small_img, caption = "Second Image")
                st.write("Second Image dimensions:", small_img.shape)

                blended = cv2.addWeighted(src1 = large_img, alpha = alpha_value, src2 = small_img, beta = beta_value, gamma = gamma_value)
                st.image(blended, caption = "After Blending")
                st.write("Result Image dimensions:", blended.shape)

        # Blending images of same size
        if blend_choice == "Blend Image of Different Size":

            #creating of columns
            col1, col2 = st.columns([1, 3])

            with col1: 
                x_offset = st.slider("Choose x of Starting Point", 0, 799, value = 0)
                y_offset = st.slider("Choose y of Starting Point", 0, 799, value = 0)

            with col2:
                large_img = cv2.imread("Image Processing Simulations/Image set/Blending or Pasting/image1.jpeg")
                large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)
                large_img = cv2.resize(large_img,(1024,1024))
                small_img = cv2.imread("Image Processing Simulations/Image set/Blending or Pasting/image2.png")
                small_img = cv2.resize(small_img, (400,400))
                small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

                st.image(large_img, caption = "Larger Image")
                st.write("Larger Image dimensions:", large_img.shape)
                st.image(small_img, caption = "Smaller Image")
                st.write("Smaller Image dimensions:", small_img.shape)
                
                # ROI
                x_end = x_offset + small_img.shape[1]
                y_end = y_offset + small_img.shape[0]
                roi = large_img[y_offset:y_end, x_offset:x_end]

                # Mask 
                small_img_gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                mask_inv = cv2.bitwise_not(small_img_gray)
                mask_inv = cv2.bitwise_not(mask_inv)

                # Blending ROI
                final_roi = cv2.bitwise_or(roi, roi, mask = mask_inv)

                x = 0
                y = 0 

                for i in range(y_offset, y_end):
                    for j in range(x_offset, x_end):
                        large_img[i, j] = final_roi[y, x]
                        if x < 400:
                            x = x+1
                    #if y < 224:
                    y = y+1
                    x = 0

                st.image(large_img, caption = "Blended Image")
                st.write("Blended Image dimensions:", large_img.shape)


    if choice == "Pasting":

        #creating of columns
        col1, col2 = st.columns([1, 3])
                
        with col1: 
            x_offset = st.slider("Choose x of Starting Point", 0, 799, value = 0)
            y_offset = st.slider("Choose y of Starting Point", 0, 799, value = 0)

        with col2:
            small_img = cv2.imread("Image Processing Simulations/Image set/Blending or Pasting/image1.jpeg")
            small_img = cv2.resize(small_img, (300,300))
            small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            large_img = cv2.imread("Image Processing Simulations/Image set/Blending or Pasting/image2.png")
            large_img = cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB)

            st.image(small_img, caption = "First Image")
            st.write("First Image dimensions:", small_img.shape)
            st.image(large_img, caption = "Second Image")
            st.write("Second Image dimensions:", large_img.shape)

            x_end = x_offset + small_img.shape[1]
            y_end = y_offset + small_img.shape[0]

            x = 0
            y = 0 

            for i in range(y_offset, y_end):
                for j in range(x_offset, x_end):
                    large_img[i, j] = small_img[y, x]
                    if x < 300:
                        x = x+1
                #if y < 224:
                y = y+1
                x = 0

            st.image(large_img, caption = "Pasted Image")
            st.write("Pasted Image dimensions:", large_img.shape)

# SIMULATION 
if main_choice == "Blurring or Smoothening":

    # title
    st.title("Run Simulation for Blurring and Smoothening of Image")

    # choice:
    choice = st.selectbox("Blurring of Image", ["Select one", "Blurring of Image", "Smoothening of Image"])

    #for Blurring
    if choice == "Blurring of Image":
        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            horizontal_blurring = st.slider("Choose degree of Horizontal Blur", 0, 100, value = 10)
            vertical_blurring = st.slider("Choose degree of Vertical Blur", 0, 100, value = 10)

        with col2:
            
            col1_1, col1_2 = st.columns([1, 1])

            with col1_1:

                img = cv2.imread("Image Processing Simulations/Image set/Blurring or Smoothening/image1.jpeg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img, caption = "Original Image")
                st.write("Original Image dimensions:", img.shape)

                dst = cv2.blur(img,ksize = (horizontal_blurring,vertical_blurring))  
                st.image(dst, caption = "Smoothened Image")
                st.write("Blurred Image dimensions:", dst.shape)

            with col1_2:

                img = cv2.imread("Image Processing Simulations/Image set/Blurring or Smoothening/image1.jpeg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img, caption = "Original Image")
                st.write("Original Image dimensions:", img.shape)

                dst = cv2.blur(img,ksize = (horizontal_blurring,vertical_blurring))  
                st.image(dst, caption = "Smoothened Image")
                st.write("Blurred Image dimensions:", dst.shape)

        #for Smootheing
    if choice == "Smoothening of Image":
        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            horizontal_smoothening = st.slider("Choose degree of Horizontal Smoothening", 0, 100, value = 100)
            vertical_smoothening = st.slider("Choose degree of Vertical Smoothening", 0, 100, value = 100)
            brightness = st.slider("Choose degree of brightness of Image", 0,100, value = 10)

        with col2:

            col1_1, col1_2 = st.columns([1, 1])

            with col1_1:

                img = cv2.imread("Image Processing Simulations/Image set/Blurring or Smoothening/image1.jpeg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img, caption = "Original Image")
                st.write("Original Image dimensions:", img.shape)

                dst = cv2.bilateralFilter(img,brightness,horizontal_smoothening,vertical_smoothening)
                st.image(dst, caption = "Smoothened Image")
                st.write("Smoothened Image dimensions:", dst.shape)

            with col1_2:

                img = cv2.imread("Image Processing Simulations/Image set/Blurring or Smoothening/image1.jpeg")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                st.image(img, caption = "Original Image")
                st.write("Original Image dimensions:", img.shape)

                dst = cv2.bilateralFilter(img,brightness,horizontal_smoothening,vertical_smoothening)
                st.image(dst, caption = "Smoothened Image")
                st.write("Smoothened Image dimensions:", dst.shape)

# Edge Detection
if main_choice == "Edge Detection":

    # title
    st.title("Run Simulation for Edge Detection")

    # choice:
    choice = st.selectbox("Edge Detection", ["Select one", "Edge Detection"])

    #for Blurring
    if choice == "Edge Detection":
        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            blockSize = st.slider("Choose thickness of edge", 0, 10, value = 5)
            k_value = st.slider("Detection of curved edges", 0.0, 10.0, value = 0.05)

        with col2:
            img = cv2.imread("Image Processing Simulations/Image set/Edge Detection/image.png")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.image(img, caption = "Original Image")
            st.write("Original Image dimensions:", img.shape)

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray_img)
            dst = cv2.cornerHarris(src=gray,blockSize=blockSize,ksize=3,k=k_value)
            #dst = cv2.dilate(dst,None)
            #img[dst>0.01*dst.max()] = [255,0,0]
            dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            img = np.float32(img) / 255.0
            blended = cv2.addWeighted(src1 = img, alpha = 0.5, src2 = dst, beta = 1, gamma = 1)

            st.image(blended,caption = "Detect Edges",clamp=True, channels='BGR')
            st.write("Result Image dimensions:", blended.shape)


# Haar Cascade - Face ,Eyes, Car Number Plate Detection
if main_choice == "Feature Detection":

    # function to convert hex_color to rgb
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    # title
    st.title("Run Simulation for Feature Detection")

    # choice:
    choice = st.selectbox("Feature To Detect", ["Select one", "Face Detection", "Eye Detection", "Car Number Plate Detection"])

    #for 
    if choice == "Face Detection":
        img_choice = st.selectbox("Choice of image", ["Select one", "Image with clear distinction", "Image with unclear distinction", "Group Image"])

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            thickness = st.slider("Thickness of rectangle", 1, 20, value = 4)
            color_shape = st.color_picker("Pick a color for the shape", key= "shape", value = "#FFFFFF")
            color_shape = hex_to_rgb(color_shape)

        with col2:
            if img_choice == "Image with clear distinction":
                face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/Nadia_Murad.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                st.image(face_img, caption = "Original Image")
                st.write("Original Image dimensions:", face_img.shape)

                face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_frontalface_default.xml")

                face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
                for (x,y,w,h) in face_rects:
                    cv2.rectangle(face_img,(x,y),(x+w,y+h), color_shape,thickness)
                
                st.image(face_img, caption = "Result")
            
            if img_choice == "Image with unclear distinction":
                face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/Denis_Mukwege.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                st.image(face_img, caption = "Original Image")
                st.write("Original Image dimensions:", face_img.shape)

                face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_frontalface_default.xml")

                face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
                for (x,y,w,h) in face_rects:
                    cv2.rectangle(face_img,(x,y),(x+w,y+h),color_shape,thickness)
                
                st.image(face_img, caption = "Result")

            if img_choice == "Group Image":
                face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/solvay_conference.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                st.image(face_img, caption = "Original Image")
                st.write("Original Image dimensions:", face_img.shape)

                face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_frontalface_default.xml")

                face_rects = face_cascade.detectMultiScale(face_img)
                for (x,y,w,h) in face_rects:
                    cv2.rectangle(face_img,(x,y),(x+w,y+h),color_shape,thickness)
                
                st.image(face_img, caption = "Result")

    #for 
    if choice == "Eye Detection":
        img_choice = st.selectbox("Choice of image", ["Select one", "Image with clear distinction", "Image with unclear distinction"])

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            thickness = st.slider("Thickness of rectangle", 1, 20, value = 4)
            color_shape = st.color_picker("Pick a color for the shape", key= "shape", value = "#FFFFFF")
            color_shape = hex_to_rgb(color_shape)

        with col2:
            if img_choice == "Image with clear distinction":
                face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/Nadia_Murad.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                st.image(face_img, caption = "Original Image")
                st.write("Original Image dimensions:", face_img.shape)

                face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_eye.xml")

                face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
                for (x,y,w,h) in face_rects:
                    cv2.rectangle(face_img,(x,y),(x+w,y+h),color_shape,thickness)
                
                st.image(face_img, caption = "Result")
            
            if img_choice == "Image with unclear distinction":
                face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/Denis_Mukwege.jpg")
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                st.image(face_img, caption = "Original Image")
                st.write("Original Image dimensions:", face_img.shape)

                face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_eye.xml")

                face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
                for (x,y,w,h) in face_rects:
                    cv2.rectangle(face_img,(x,y),(x+w,y+h),color_shape,thickness)
                
                st.image(face_img, caption = "Result")

        #for 
    if choice == "Car Number Plate Detection":
        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            thickness = st.slider("Thickness of rectangle", 1, 10, value = 4)
            color_shape = st.color_picker("Pick a color for the shape", key= "shape", value = "#0000FF")
            color_shape = hex_to_rgb(color_shape)

        with col2:
            face_img = cv2.imread("Image Processing Simulations/Image set/Feature Detection/car_plate.jpg")
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            st.image(face_img, caption = "Original Image")
            st.write("Original Image dimensions:", face_img.shape)

            face_cascade = cv2.CascadeClassifier("Image Processing Simulations/Image set/Feature Detection/haarcascade_russian_plate_number.xml")

            face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)
            for (x,y,w,h) in face_rects:
                cv2.rectangle(face_img,(x,y),(x+w,y+h),color_shape,thickness)
            
            st.image(face_img, caption = "Result")


# Watershed Algorithm
if main_choice == "Watershed Algorithm":

    # title
    st.title("Run Simulation to Demonstrate Watershed Algorithm")

    col1, col2 = st.columns([1, 3])

    with col2:
        img = cv2.imread("Image Processing Simulations/Image set/Watershed Algorithm/image1.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        st.image(img, caption = "Original Image")
        st.write("Original Image dimensions:", img.shape)

        mask = np.zeros(img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (50,50,450,290)
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]

        st.image(img, "Front of Image")
        st.write("Front of Image", img.shape)

# Feature Matching
if main_choice == "Feature Matching":

    # title
    st.title("Run Simulation to Demonstrate 2 methods of Feature Detection")

    # choice:
    choice = st.selectbox("Method to Demonstrate", ["Select one", "Brute-Force Matching with ORB Descriptors", "FLANN based Matcher"])

    #for 
    if choice == "Brute-Force Matching with ORB Descriptors":

        template_choice = st.selectbox("Select Image for features", ["Select one", "Image1", "Image2"])

        if template_choice == "Image1":
            reeses = cv2.imread("Image Processing Simulations/Image set/Feature Matching/lucky_charms.jpg")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)

            st.image(reeses, caption = "Template Image")
            st.write("Template Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Feature Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Main Image")
            st.write("Main Image dimensions:", cereals.shape)

            orb = cv2.ORB_create()

            kp1, des1 = orb.detectAndCompute(reeses, None)  
            kp2, des2 = orb.detectAndCompute(cereals, None)  

            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)   
            matches = bf.match(des1,des2)
            single_match = matches[0]
            single_match.distance
            matches = sorted(matches,key=lambda x:x.distance)
            reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

            st.image(reeses_matches, caption = "Features Matched")
            st.write("Result Image dimensions:", reeses_matches.shape)

        if template_choice == "Image2":
            reeses = cv2.imread("Image Processing Simulations/Image set/Feature Matching/reeses_puffs.png")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)

            st.image(reeses, caption = "Template Image")
            st.write("Template Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Feature Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Main Image")
            st.write("Main Image dimensions:", cereals.shape)

            orb = cv2.ORB_create()

            kp1, des1 = orb.detectAndCompute(reeses, None)  
            kp2, des2 = orb.detectAndCompute(cereals, None)  

            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)   
            matches = bf.match(des1,des2)
            single_match = matches[0]
            single_match.distance
            matches = sorted(matches,key=lambda x:x.distance)
            reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)

            st.image(reeses_matches, caption = "Features Matched")
            st.write("Result Image dimensions:", reeses_matches.shape)


    #for 
    if choice == "FLANN based Matcher":

        template_choice = st.selectbox("Select Image for features", ["Select one", "Image1", "Image2"])

        if template_choice == "Image1":
            reeses = cv2.imread("Image Processing Simulations/Image set/Feature Matching/lucky_charms.jpg")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)

            st.image(reeses, caption = "Template Image")
            st.write("Template Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Feature Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Main Image")
            st.write("Main Image dimensions:", cereals.shape)

            sift = cv2.SIFT_create()

            kp1, des1 = sift.detectAndCompute(reeses, None)  
            kp2, des2 = sift.detectAndCompute(cereals, None)  

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test
            for i,(match1,match2) in enumerate(matches):
                if match1.distance < 0.7*match2.distance:
                    matchesMask[i]=[1,0]

            draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = matchesMask,
                            flags = 0)

            flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)

            st.image(flann_matches, caption = "Features Matched")
            st.write("Result Image dimensions:", flann_matches.shape)

        if template_choice == "Image2":
            reeses = cv2.imread("Image Processing Simulations/Image set/Feature Matching/reeses_puffs.png")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)

            st.image(reeses, caption = "Template Image")
            st.write("Template Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Feature Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Main Image")
            st.write("Main Image dimensions:", cereals.shape)

            sift = cv2.SIFT_create()

            kp1, des1 = sift.detectAndCompute(reeses, None)  
            kp2, des2 = sift.detectAndCompute(cereals, None)  

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test
            for i,(match1,match2) in enumerate(matches):
                if match1.distance < 0.7*match2.distance:
                    matchesMask[i]=[1,0]

            draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = matchesMask,
                            flags = 0)

            flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)

            st.image(flann_matches, caption = "Features Matched")
            st.write("Result Image dimensions:", flann_matches.shape)


# Template Matching
if main_choice == "Template Matching":

    # function to convert hex_color to rgb
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    # title
    st.title("Run Simulation for Template Detection")

    # choice:
    choice = st.selectbox("Template To Detect", ["Select one", "Instance1", "Instance2"])

    #for 
    if choice == "Instance1":

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            color_shape = st.color_picker("Pick a color for the shape", key= "shape", value = "#00FF00")
            color_shape = hex_to_rgb(color_shape)
            thickness = st.slider("Thickness", 1, 20, value = 4)

        with col2:

            reeses = cv2.imread("Image Processing Simulations/Image set/Template Matching/reeses_puffs.png")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)
            reeses = cv2.resize(reeses,(200,200))

            st.image(reeses, caption = "Original Image")
            st.write("Original Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Template Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Original Image")
            st.write("Original Image dimensions:", cereals.shape)

            methods = ['cv2.TM_SQDIFF']

            height, width, channels = reeses.shape

            for m in methods:
                
                # Get the actual function instead of the string
                method = eval(m)

                # Apply template Matching with the method
                res = cv2.matchTemplate(cereals,reeses,method)
                
                # Grab the Max and Min values, plus their locations
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # Set up drawing of Rectangle
                
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                # Notice the coloring on the last 2 left hand side images.
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc    
                else:
                    top_left = max_loc
                    
                # Assign the Bottom Right of the rectangle
                bottom_right = (top_left[0] + width, top_left[1] + height)

                # Draw the Red Rectangle
                cv2.rectangle(cereals,top_left, bottom_right, color_shape, 10)

                st.image(cereals)
            

    if choice == "Instance2":

        #creating of columns
        col1, col2 = st.columns([1, 3])
        
        with col1:
            color_shape = st.color_picker("Pick a color for the shape", key= "shape", value = "#00FF00")
            color_shape = hex_to_rgb(color_shape)
            thickness = st.slider("Thickness", 1, 20, value = 4)

        with col2:

            reeses = cv2.imread("Image Processing Simulations/Image set/Template Matching/lucky_charms.jpg")
            reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)
            reeses = cv2.resize(reeses,(200,200))

            st.image(reeses, caption = "Original Image")
            st.write("Original Image dimensions:", reeses.shape)

            cereals = cv2.imread("Image Processing Simulations/Image set/Template Matching/many_cereals.jpg")
            cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

            st.image(cereals, caption = "Original Image")
            st.write("Original Image dimensions:", cereals.shape)

            methods = ['cv2.TM_SQDIFF']

            height, width, channels = reeses.shape

            for m in methods:
                
                # Get the actual function instead of the string
                method = eval(m)

                # Apply template Matching with the method
                res = cv2.matchTemplate(cereals,reeses,method)
                
                # Grab the Max and Min values, plus their locations
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # Set up drawing of Rectangle
                
                # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                # Notice the coloring on the last 2 left hand side images.
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc    
                else:
                    top_left = max_loc
                    
                # Assign the Bottom Right of the rectangle
                bottom_right = (top_left[0] + width, top_left[1] + height)

                # Draw the Red Rectangle
                cv2.rectangle(cereals,top_left, bottom_right, color_shape, 10)

                st.image(cereals)

        # if choice == "Instance3":

        #     #creating of columns
        #     col1, col2 = st.columns([1, 3])
            
        #     with col1:
        #         thickness = st.slider("Thickness", 1, 20, value = 4)

        #     with col2:

        #         reeses = cv2.imread("VirtualLab_ImageProcessing/Experiment9/sammy_face.jpg")
        #         reeses = cv2.cvtColor(reeses, cv2.COLOR_BGR2RGB)
        #         reeses = cv2.resize(reeses,(200,200))

        #         st.image(reeses, caption = "Original Image")
        #         st.write("Original Image dimensions:", reeses.shape)

        #         cereals = cv2.imread("VirtualLab_ImageProcessing/Experiment9/sammy.jpg")
        #         cereals = cv2.cvtColor(cereals, cv2.COLOR_BGR2RGB)

        #         st.image(cereals, caption = "Original Image")
        #         st.write("Original Image dimensions:", cereals.shape)

        #         methods = ['cv2.TM_CCOEFF']

        #         height, width, channels = reeses.shape

        #         for m in methods:
                    
        #             # Get the actual function instead of the string
        #             method = eval(m)

        #             # Apply template Matching with the method
        #             res = cv2.matchTemplate(cereals,reeses,method)
                    
        #             # Grab the Max and Min values, plus their locations
        #             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    
        #             # Set up drawing of Rectangle
                    
        #             # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        #             # Notice the coloring on the last 2 left hand side images.
        #             if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #                 top_left = min_loc    
        #             else:
        #                 top_left = max_loc
                        
        #             # Assign the Bottom Right of the rectangle
        #             bottom_right = (top_left[0] + width, top_left[1] + height)

        #             # Draw the Red Rectangle
        #             cv2.rectangle(cereals,top_left, bottom_right, (0,255,0), 10)

        #             st.image(cereals)



# Image Inprinting
if main_choice == "Image Inprinting":

    # title
    st.title("Run Simulation for Image Inprinting")


    #creating of columns
    col1, col2 = st.columns([1, 1])
    
    with col2:
        img = cv2.imread("Image Processing Simulations/Image set/Image Inprinting/cat_damaged.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption = "Original Image")
        st.write("Original Image dimensions:", img.shape)

        mask = cv2.imread("Image Processing Simulations/Image set/Image Inprinting/cat_mask.png",0)

        st.image(mask, caption = "Mask Image")
        st.write("Mask Image dimensions:", mask.shape)

        dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)   

        st.image(dst, caption= "Enhanced Image")
        st.write("Enhanced Image dimensions:", dst.shape)

    with col1:
        img = cv2.imread("Image Processing Simulations/Image set/Image Inprinting/messi.jpeg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(400,400))
        st.image(img, caption = "Original Image")
        st.write("Original Image dimensions:", img.shape)

        mask = cv2.imread("Image Processing Simulations/Image set/Image Inprinting/messi_mask.jpeg" ,0)
        mask = cv2.resize(mask,(400,400))

        st.image(mask, caption = "Mask Image")
        st.write("Mask Image dimensions:", mask.shape)

        dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)   

        st.image(dst)

