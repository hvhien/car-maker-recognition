# from tensorflow.keras.models import load_model,Model
# from flask import Flask,render_template
#
#
# app=Flask(__name__)
#
# @app.route('/',methods=['GET','POST'])
# def home():
#     return render_template('index.html')
#
#
# if __name__=="__main__":
#     app.run()


from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st
st.title("Nhận diện hãng sản xuất xe oto")
model=load_model('car.h5')
st.set_option('deprecation.showfileUploaderEncoding',False)
label_map={'AM': 0,'Acura': 1,'Aston': 2,'Audi': 3,'BMW': 4,'Bentley': 5,'Bugatti': 6,'Buick': 7,'Cadillac': 8,'Chevrolet': 9,'Chrysler': 10,
           'Daewoo': 11,'Dodge': 12,'Eagle': 13,'FIAT': 14,'Ferrari': 15,'Fisker': 16,'Ford': 17,'GMC': 18,'Geo': 19,'HUMMER': 20,'Honda': 21,
           'Hyundai': 22,'Infiniti': 23,'Isuzu': 24,'Jaguar': 25,'Jeep': 26,'Lamborghini': 27,'Land': 28,'Lincoln': 29,'MINI': 30,'Maybach': 31,
           'Mazda': 32,'McLaren': 33,'Mercedes-Benz': 34,'Mitsubishi': 35,'Nissan': 36,'Plymouth': 37,'Porsche': 38,'Ram': 39,'Rolls-Royce': 40,
           'Scion': 41,'Spyker': 42,'Suzuki': 43,'Tesla': 44,'Toyota': 45,'Volkswagen': 46,'Volvo': 47,'smart': 48}
def imageProcessing(img):
    # img=Image.open(img)
    img=img.resize(size=(256,256))
    img=np.asarray(img)
    img=img/255.
    img=np.expand_dims(img,0)
    return img

def print_pred(pred):
    t = 0
    re = {}
    for i in label_map:

        re[i] = pred[0][t]
        t = t + 1
    result= dict(sorted(re.items(), key=lambda x: x[1], reverse=True))
    df = pd.DataFrame.from_dict(result, orient='index', columns=['acc'])
    df = df.reset_index()
    df=df.sort_values(by='acc',ascending=False)
    st.dataframe(df)
    return "Đây là xe %s với độ chính xác dự đoán là %f%"%(df['index'][0],df['acc'][0])
file=st.file_uploader(" ",type=["jpg","png","jpeg"])
if st.button("Nhấn vào đây để xem hãng sản xuất"):
    if file is None:
        st.error("Ôi bạn ơi! bạn phải nhập ảnh đã chứ")
    else:
        img=Image.open(file)
        st.image(img, use_column_width=False, width=300)

        img=imageProcessing(img)

        pred=model.predict(img)

        st.header(print_pred(pred*100))
st.write("hello")