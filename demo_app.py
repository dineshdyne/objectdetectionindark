import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import keras
from keras.models import load_model
from numpy.lib.polynomial import poly

st.set_page_config(  # Alternate names: setup_page, page, layout
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="wide",
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title=f"Low Light Object Detection",
    page_icon=None,  # String, anything supported by st.image, or None.
)
image = Image.open("images/sandy.jpg")

image = image.resize((200, 200), Image.NEAREST)
st.sidebar.image(image, use_column_width=False)
st.sidebar.title(f"Low Light Object Detection")


f = st.file_uploader("Upload File", type=['JPG', 'PNG'])


# img = cv2.imread('C:/Users/dinesh/PycharmProjects/pythonProject/test1.png')
# img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10,10))
# plt.axis('off')
# box, label, count = cv.detect_common_objects(img)
# output = draw_bbox(img, box, label, count)
# output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 10))
# plt.axis("off")
# plt.imshow(output)
# plt.show()
# print("Number of objects in this image are " + str(len(label)))
def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright


def blur_image(image, amount):
    img = cv2.cvtColor(image, 1)
    blur_img = cv2.GaussianBlur(img, (11, 11), amount)
    return blur_img


def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

@st.cache(
    allow_output_mutation=True, suppress_st_warning=True, max_entries=None, ttl=None
)
def load_saved_model():
    model=load_model('main_model')
    return model
model=load_saved_model()
if f is not None:

    # uploaded_image = Image.open(f)
    # img_1= keras.preprocessing.image.img_to_array(uploaded_image ).astype(int)
    # st.image(img_1)
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, 1)[:,:,::-1]
    st.image(img)
    #st.write(img==img_1)

    img=cv2.resize(img, (256, 256),interpolation=cv2.INTER_NEAREST).astype(int)
    #st.write(type(img),img[0])
    enhance=st.checkbox("Enhance")
    brightness=st.slider("Brightness Scale Increase",min_value=1,max_value=100,value=1,step=1)
    col1,col2=st.columns([1,1])
    # Now do something with the image! For example, let's display it:
    col1.image(np.array(img))

    if enhance:
        img=enhance_details(img)
    img=brighten_image(img,brightness)
    col2.image(img)
    k=img.copy()
    # st.image(img)
    st.write(f"sbjhvnz {k.shape}")
    box, label, perc = cv.detect_common_objects(img)
    output = draw_bbox(img, box, label, perc)
    st.write('Hi')
    st.image(k)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.image(output)

    st.write("Done")
    # st.write(label, perc)
    st.write(f"No of Detections: {len(label)}")

    st.dataframe(pd.DataFrame(zip(label, perc),
                 columns=['Labels', 'confidence']))
    

    
    image = np.expand_dims(k.astype('float32') / 255.0, axis=0)
    st.image(image)
    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0,255)
    st.write(output_image.shape,np.min(output_image),np.max(output_image))
    output_image = output_image.reshape((np.shape(output_image)[0],np.shape(output_image)[1],3))
    output_image = output_image.astype('uint8')
    st.write(output_image.dtype)
    
    st.image(output_image)
    st.image(Image.fromarray(output_image.astype('uint8'),'RGB'))
    

    box, label, perc = cv.detect_common_objects(output_image)
    output = draw_bbox(output_image, box, label, perc)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    st.image(output_image)
    st.dataframe(pd.DataFrame(zip(label, perc),
                 columns=['Labels', 'confidence']))
    # final=Image.fromarray(output_image.astype('uint8'))
    
    # st.image(final,channels='BGR')
